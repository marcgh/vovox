
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>

#include <VX/vx.h>
#include <NVX/nvx_timer.hpp>

#include "NVXIO/FrameSource.hpp"
#include "NVXIO/Render.hpp"
#include "NVXIO/Application.hpp"
#include "NVXIO/Utility.hpp"


int main()
{
    std::cout << "Hello World" << std::endl;

    nvxio::Application &app = nvxio::Application::get();

    return 0;
}



