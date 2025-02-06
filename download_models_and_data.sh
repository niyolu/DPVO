#!/bin/bash

if [ ! -f "dpvo.pth" ]; then
    wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip && unzip models.zip
fi