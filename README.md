# Installation

Assuming you have python3.7 installed (otherwise adapt the filename in the last line of the instructions accordingly)


  git clone --recursive 
  cd alpha-particles-cpp
  mkdir build
  cd build
  cmake -DPYTHON_EXECUTABLE:FILEPATH=python3 ..
  cd ../drivers
  ln -s ../build/pyparticle.cpython-37m-darwin.so

Then run
  
  python3 driver.py
