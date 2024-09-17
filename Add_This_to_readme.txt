The error message indicates that the CUDA installer failed to verify the gcc (GNU Compiler Collection) version. This is a common issue when installing CUDA. Here's what you can do to resolve this:

1. First, let's check your current gcc version:

   ```
   gcc --version
   ```

2. If gcc is not installed or the version is not compatible, you'll need to install or update it. CUDA 12.4 typically requires gcc version 11 or lower. Let's install gcc-11:

   ```
   sudo apt update
   sudo apt install gcc-11 g++-11
   ```

3. After installation, set gcc-11 as the default:

   ```
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11
   sudo update-alternatives --config gcc
   ```

   Choose the option for gcc-11 if prompted.

4. Verify the gcc version again:

   ```
   gcc --version
   ```

5. Now, try running the CUDA installer again, but this time with some additional options to ignore the gcc version check:

   ```
   sudo sh cuda_12.4.0_550.54.14_linux.run --override
   ```

6. If you still encounter issues, you can try running the installer with more verbose output:

   ```
   sudo sh cuda_12.4.0_550.54.14_linux.run --verbose
   ```

   This will provide more detailed information about any issues encountered during installation.

7. If the problem persists, check the log file mentioned in the error message:

   ```
   sudo cat /var/log/cuda-installer.log
   ```

   This log might provide more specific information about why the gcc verification is failing.

8. As a last resort, if you're still having trouble, you can try forcing the installation to proceed without the gcc check:

   ```
   sudo sh cuda_12.4.0_550.54.14_linux.run --silent --override --toolkit --samples --samplespath=/usr/local/cuda-12.4/samples
   ```

   Be cautious with this approach, as it bypasses some compatibility checks.

After successful installation, don't forget to update your PATH and LD_LIBRARY_PATH as mentioned in the previous instructions.

If you continue to face issues, please provide the output of the `gcc --version` command and any relevant parts of the `/var/log/cuda-installer.log` file, and I'll be happy to help you further.