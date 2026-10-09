# Third-Party Notices

The Piper WebXR teleoperation implementation was informed by the following projects:

- **LeRobot / Hugging Face** — Apache License 2.0. The recording schema and camera integration use the installed LeRobot APIs.
- **UFACTORY Piper SDK and local Piper adapter reference** — Apache License 2.0 where marked. The StarVLA adapter is an independent, minimal implementation against `piper_sdk`.
- **A-Frame 1.7.1** — MIT License. `deployment/teleoperation/vr/web_ui/vendor/aframe.min.js` is redistributed unchanged from the official A-Frame release CDN.
- **Roboto MSDF font assets** — Apache License 2.0. The local WebXR status text uses the A-Frame CDN's Roboto MSDF atlas; its license is bundled at `deployment/teleoperation/vr/web_ui/vendor/fonts/LICENSE.roboto.txt`.
- **LogosVLA deployment WebXR implementation** — used as a behavioral reference for controller collection, relative pose anchoring, and operator events. Existing source headers and upstream notices must be retained if code is copied verbatim.
- **Lerobot_Pika_Piperx_Cam host-side Piper IK** — Apache License 2.0. StarVLA's `official_kinematics.py` is a modified, in-process adaptation of the Pinocchio/CasADi Piper solver from commit `1e706b02af8006ad2ffd2abf288743d979f69382`; the LeRobot/Pika wrappers are not redistributed.
- **AgileX PikaAnyArm Piper IK** — BSD 3-Clause License, Copyright (c) 2020 Tixiao Shan. The downstream host-IK implementation was originally adapted from PikaAnyArm's `forward_inverse_kinematics.py`.
- **AgileX Piper-X URDF** — MIT License, Copyright (c) 2026 aalicecc. The packaged model subset comes from `agilexrobotics/agx_arm_urdf` commit `f6642ce0d7872c686f29c99e9e10cd23d1d49313`; its license and provenance are under `deployment/teleoperation/piper/assets/agx_arm_description/`.
- **Pinocchio** — BSD 2-Clause License; **CasADi** — LGPL-3.0-or-later; **IPOPT** — EPL-2.0. These are installed runtime dependencies for host-side IK and are not vendored by this source tree.

These notices supplement, and do not replace, the license terms in the corresponding third-party distributions.
