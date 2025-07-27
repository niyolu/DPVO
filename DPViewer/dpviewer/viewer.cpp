#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <iostream>
#include <mutex>

#include "viewer_cuda.h"

typedef unsigned char uchar;

// Forward declaration from viewer_cuda.h, assuming its existence
torch::Tensor poseToMatrix(const torch::Tensor poses);

class Viewer {
public:
    Viewer(
        const torch::Tensor image,
        const torch::Tensor poses,
        const torch::Tensor points,
        const torch::Tensor colors,
        const torch::Tensor intrinsics);

    // This is now a blocking call that runs the entire GUI loop
    void run();
    void update_image(torch::Tensor img);

private:
    void drawPoints();
    void drawPoses();
    void initVBO();
    void destroyVBO();

    torch::Tensor image;
    torch::Tensor poses;
    torch::Tensor points;
    torch::Tensor colors;
    torch::Tensor intrinsics;

    int w, h;
    int nFrames, nPoints;
    bool redraw;
    torch::Tensor transformMatrix;
    std::mutex mtx;

    GLuint vbo, cbo;
    struct cudaGraphicsResource *xyz_res;
    struct cudaGraphicsResource *rgb_res;
};

Viewer::Viewer(
    const torch::Tensor image,
    const torch::Tensor poses,
    const torch::Tensor points,
    const torch::Tensor colors,
    const torch::Tensor intrinsics)
    : image(image), poses(poses), points(points), colors(colors), intrinsics(intrinsics) {
    redraw = true;
    h = image.size(0);
    w = image.size(1);
    nFrames = poses.size(0);
    nPoints = points.size(0);
}

void Viewer::update_image(torch::Tensor img) {
    std::lock_guard<std::mutex> lock(mtx);
    redraw = true;
    this->image = img.permute({1, 2, 0}).to(torch::kCPU);
}

void Viewer::run() {
    cudaSetDevice(0);
    pangolin::CreateWindowAndBind("DPVO", 1280, 960);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, -2, -2, 0, 0, 0, pangolin::AxisNegY));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0, -w / (float)h)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::GlTexture texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::View& d_video = pangolin::Display("imgVideo").SetAspect(w / (float)h);
    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, 0.0, 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_video);

    initVBO();

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        d_cam.Activate(s_cam);
        
        transformMatrix = poseToMatrix(poses).transpose(1, 2).contiguous().to(torch::kCPU);
        drawPoints();
        drawPoses();

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (redraw) {
                texVideo.Upload(image.data_ptr(), GL_BGR, GL_UNSIGNED_BYTE);
                redraw = false;
            }
        }

        d_video.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        texVideo.RenderToViewportFlipY();
        pangolin::FinishFrame();
    }

    destroyVBO();
}

void Viewer::drawPoints() {
  float *xyz_ptr;
  uchar *rgb_ptr;
  size_t xyz_bytes;
  size_t rgb_bytes; 

  unsigned int size_xyz = 3 * points.size(0) * sizeof(float);
  unsigned int size_rgb = 3 * points.size(0) * sizeof(uchar);

  cudaGraphicsResourceGetMappedPointer((void **) &xyz_ptr, &xyz_bytes, xyz_res);
  cudaGraphicsResourceGetMappedPointer((void **) &rgb_ptr, &rgb_bytes, rgb_res);

  float *xyz_data = points.data_ptr<float>();
  cudaMemcpy(xyz_ptr, xyz_data, xyz_bytes, cudaMemcpyDeviceToDevice);

  uchar *rgb_data = colors.data_ptr<uchar>();
  cudaMemcpy(rgb_ptr, rgb_data, rgb_bytes, cudaMemcpyDeviceToDevice);

  // bind color buffer
  glBindBuffer(GL_ARRAY_BUFFER, cbo);
  glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(3, GL_FLOAT, 0, 0);

  // bind vertex buffer
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, points.size(0));
  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glDisableClientState(GL_COLOR_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void Viewer::drawPoses() {

  float *tptr = transformMatrix.data_ptr<float>();

  const int NUM_POINTS = 8;
  const int NUM_LINES = 10;

  const float CAM_POINTS[NUM_POINTS][3] = {
    { 0,   0,   0},
    {-1,  -1, 1.5},
    { 1,  -1, 1.5},
    { 1,   1, 1.5},
    {-1,   1, 1.5},
    {-0.5, 1, 1.5},
    { 0.5, 1, 1.5},
    { 0, 1.2, 1.5}};

  const int CAM_LINES[NUM_LINES][2] = {
    {1,2}, {2,3}, {3,4}, {4,1}, {1,0}, {0,2}, {3,0}, {0,4}, {5,7}, {7,6}};

  const float SZ = 0.05;

  glColor3f(0,0.5,1);
  glLineWidth(1.5);

  for (int i=0; i<nFrames; i++) {

      if (i + 1 == nFrames)
        glColor3f(1,0,0);

      glPushMatrix();
      glMultMatrixf((GLfloat*) (tptr + 4*4*i));

      glBegin(GL_LINES);
      for (int j=0; j<NUM_LINES; j++) {
        const int u = CAM_LINES[j][0], v = CAM_LINES[j][1];
        glVertex3f(SZ*CAM_POINTS[u][0], SZ*CAM_POINTS[u][1], SZ*CAM_POINTS[u][2]);
        glVertex3f(SZ*CAM_POINTS[v][0], SZ*CAM_POINTS[v][1], SZ*CAM_POINTS[v][2]);
      }
      glEnd();

      glPopMatrix();
  }
}

void Viewer::initVBO() {
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  // initialize buffer object
  unsigned int size_xyz = 3 * points.size(0) * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size_xyz, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&xyz_res, vbo, cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &xyz_res, 0);

  glGenBuffers(1, &cbo);
  glBindBuffer(GL_ARRAY_BUFFER, cbo);

  // initialize buffer object
  unsigned int size_rgb = 3 * points.size(0) * sizeof(uchar);
  glBufferData(GL_ARRAY_BUFFER, size_rgb, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&rgb_res, cbo, cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &rgb_res, 0);
}

void Viewer::destroyVBO() {
  cudaGraphicsUnmapResources(1, &xyz_res, 0);
  cudaGraphicsUnregisterResource(xyz_res);
  glBindBuffer(1, vbo);
  glDeleteBuffers(1, &vbo);

  cudaGraphicsUnmapResources(1, &rgb_res, 0);
  cudaGraphicsUnregisterResource(rgb_res);
  glBindBuffer(1, cbo);
  glDeleteBuffers(1, &cbo);
}


// PYBIND11 MODULE
namespace py = pybind11;

PYBIND11_MODULE(dpviewerx, m) {
    py::class_<Viewer>(m, "Viewer")
        .def(py::init<const torch::Tensor, const torch::Tensor, const torch::Tensor, const torch::Tensor, const torch::Tensor>())
        .def("run", &Viewer::run) // Expose the blocking run method
        .def("update_image", &Viewer::update_image);
}