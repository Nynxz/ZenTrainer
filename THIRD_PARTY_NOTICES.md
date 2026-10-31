# Third-party notices

ZenTrainer is MIT-licensed (see [LICENSE](LICENSE)). Parts of it come from other projects under
their own licences, listed here with the notices those licences ask for.

## Fizgig

- **Used in:** `engine/zentrainer/harness/kernels/w8a16_backward.py`, the fused W8A16 Triton
  backward GEMM for int8 ConvRot weights. The kernel, its autotune configs and wrapper are by
  @mabseyuk; it was ported from Fizgig, and the file's header lists what was changed here.
- **Source:** https://github.com/shootthesound/Fizgig
- **Licence:** Apache License 2.0, below.
- **Notice:**

  ```
  Fizgig, Copyright 2026 Peter Neill
  ```

## ai-toolkit

- **Used in:** `engine/zentrainer/harness/optim/automagic3.py` and `automagic4.py`, the Automagic v3
  and v4 optimizers, vendored unchanged below a ZenTrainer header; `automagic3_batched.py` and
  `automagic4_batched.py`, the same steps run on stacks of same-shaped tensors; and `FLEX_WEIGHTS` in
  `engine/zentrainer/harness/objective.py`, averaged down from ai-toolkit's timestep weighting.
- **Source:** https://github.com/ostris/ai-toolkit
- **Licence:** MIT License, below.
- **Notice:**

  ```
  Copyright (c) 2024 Ostris, LLC
  ```

## VGL

- **Used in:** `engine/zentrainer/datasets/vgl/`, Fibo 2.0's structured-caption schema and the
  prompts that write it (commit 5eeecc2), run on a local model instead of VGL's API models.
- **Source:** https://github.com/Bria-AI/vgl
- **Licence:** MIT License, below (also in that folder's LICENSE).
- **Notice:**

  ```
  Copyright (c) 2025 BRIA AI
  ```

## InsightFace

- **Used in:** `engine/zentrainer/analysis/faces.py`, the face pipeline as InsightFace runs it
  (SCRFD's anchor decoding and overlap suppression, alignment on the five keypoints to ArcFace's
  template), and `engine/zentrainer/analysis/nets/scrfd.py` and `iresnet.py`, the SCRFD-10G and
  ArcFace IResNet networks, written to load InsightFace's exported weights.
- **Source:** https://github.com/deepinsight/insightface
- **Licence:** the code is under the MIT License, below. InsightFace's pretrained models are for
  non-commercial research use only; they aren't part of ZenTrainer and are downloaded the first
  time faces run (see "Downloaded when needed" below).
- **Notice:**

  ```
  Copyright (c) Jiankang Deng, Jia Guo and the InsightFace contributors
  ```

## OpenCV

- **Used in:** `engine/zentrainer/analysis/faces.py`, which reproduces OpenCV's resize and affine
  warp (so the aligned faces match InsightFace's to the byte) and decodes YuNet's outputs as
  OpenCV's `FaceDetectorYN` does; `engine/zentrainer/analysis/nets/yunet.py`, the YuNet network
  (opencv_zoo's face_detection_yunet_2023mar), written to load its exported weights.
- **Source:** https://github.com/opencv/opencv and https://github.com/opencv/opencv_zoo
- **Licence:** OpenCV is under the Apache License 2.0, below; the YuNet model is under the MIT
  License, below.
- **Notice:**

  ```
  OpenCV: Copyright (C) 2000-2022, Intel Corporation; 2009-2011, Willow Garage Inc.;
  2009-2016, NVIDIA Corporation; 2010-2013, Advanced Micro Devices, Inc.; 2015-2023, OpenCV
  Foundation; 2008-2016, Itseez Inc.; 2019-2023, Xperience AI; 2019-2022, Shenzhen Institute of
  Artificial Intelligence and Robotics for Society; 2022-2023, Southern University of Science
  And Technology; 2023-2025, OpenCV AI. All rights reserved.
  YuNet: Copyright (c) 2020 Shiqi Yu
  ```

## Downloaded when needed

These aren't part of ZenTrainer's code or packages: it downloads them the first time a feature
uses them, and each keeps its own licence.

- **Face models** (likeness scores, face reports), at a pinned revision, converted to safetensors in
  https://huggingface.co/nynxz/Zentrainer, which lists each file's source and licence:
  InsightFace's SCRFD-10G and ArcFace models (non-commercial research use only), and for
  commercial use OpenCV's YuNet (MIT) and fal's AuraFace (Apache License 2.0).
- **Training adapters** offered in the run form (Ostris's and others'), from their own Hugging
  Face repositories under their own licences.
- **Model files** a recipe or the model catalog offers to download, from their publishers, under
  their own licences.

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Apache License 2.0

```

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
