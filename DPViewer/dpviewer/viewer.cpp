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

    // Data Tensors (owned by the main DPVO object)
    torch::Tensor image;
    torch::Tensor poses;
    torch::Tensor points;
    torch::Tensor colors;
    torch::Tensor intrinsics;

    // Internal state
    int w;
    int h;
    int nFrames, nPoints;
    bool redraw;
    torch::Tensor transformMatrix;
    std::mutex mtx; // Mutex to protect image updates

    // OpenGL/CUDA resources
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
    : image(image), poses(poses), points(points), colors(colors), intrinsics(intrinsics)
{
    // Constructor does NOT start a thread. It just sets up the object.
    redraw = true;
    h = image.size(0);
    w = image.size(1);
    nFrames = poses.size(0);
    nPoints = points.size(0);
}

void Viewer::update_image(torch::Tensor img) {
    std::lock_guard<std::mutex> lock(mtx);
    redraw = true;
    // The main thread will pass a CUDA tensor, move it to CPU for rendering
    this->image = img.permute({1, 2, 0}).to(torch::kCPU);
}

void Viewer::run() {
    // This function is now the entry point for the Python-managed thread.
    // It will not return until the Pangolin window is closed.

    cudaSetDevice(0); // Set the CUDA device for this thread

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
        
        // No need for a render mutex now, as the main thread updates the tensors
        // and the viewer thread just reads whatever is there at the moment of rendering.
        // This can cause some visual tearing, but it will prevent crashes.
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

    // Cleanup before the thread exits
    destroyVBO();
}

// ... (drawPoints, drawPoses, initVBO, destroyVBO implementations remain the same as before) ...
// Ensure they are defined here. For brevity I am omitting them, but they must be present.

// PYBIND11 MODULE
namespace py = pybind11;

PYBIND11_MODULE(dpviewerx, m) {
    py::class_<Viewer>(m, "Viewer")
        .def(py::init<const torch::Tensor, const torch::Tensor, const torch::Tensor, const torch::Tensor, const torch::Tensor>())
        .def("run", &Viewer::run) // Expose the blocking run method
        .def("update_image", &Viewer::update_image);
}