<p align="center">
  <img src="docs/logo.png" height="240" alt="玄松 EquivFusion Logo">
</p>
<h2 align="center">EquivFusion from 玄松</h2>
<p align="center"><em>深理如松 · 验证无声</em></p>
<p align="center"><em>Slient as Pine, Precise as Logic</em></p>

# EquivFusion
EquivFusion: Unifying Formal Verification from Algorithms to Netlists for High-Efficiency Signoff.

# Contributors
- Min Li
- Baoqiz
- Mengxia Tao   <taomengxia@nctieda.com>

# Build
Linux/MacOS
```bash
cd EquivFusion
mkdir build
cd build
cmake .. -G Ninja
ninja
```
# Dependencies

- **readline>=8.2** 若readline库的头文件或者库文件不在系统搜索路径中，在执行cmake时可以通过`-DREADLINE_INCLUDE_ABSOLUTE_DIRECTORY`指定存放readline文件夹的目录的绝对路径，readline文件夹中存放的是readline相关的头文件。通过`-DREADLINE_LIBRARY_ABSOLUTE_PATH`指定readline库文件的绝对路径。


