utvisioncomponents
===============
This is the utvisioncomponents Ubitrack submodule.

Description
----------
The utvisioncomponents contains components working on camera images (based on utvision methods). Also contains components to capture and transfer images and videos.


Usage
-----
In order to use it, you have to clone the buildenvironment, change to the ubitrack directory and add the utvisioncomponents by executing:

    git submodule add https://github.com/Ubitrack/utvisioncomponents.git modules/utvisioncomponents


Dependencies
----------
In addition, this module has to following submodule dependencies which have to be added for successful building:

<table>
  <tr>
    <th>Component</th><th>Dependency</th>
  </tr>
  <tr>
    <td>all</td><td>utDataflow, utVision</td>
  </tr>
  <tr>
    <td>ImageTrigger, FrameBuffer, ImageGate, FrameSampler</td><td>utComponents</td>
  </tr>
</table>
