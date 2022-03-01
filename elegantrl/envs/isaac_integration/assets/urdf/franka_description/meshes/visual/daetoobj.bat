SET PATH=%PATH%;C:/Tools/Assimp/bin/x64/
forfiles /m *.dae /c "cmd /c assimp export @file @fname.obj --verbose --show-log -ptv"

REM SET PATH=%PATH%;C:/Program Files/VCG/MeshLab/
REM forfiles /m *.dae /c "cmd /c meshlabserver -i @file -o @fname.obj -m vn vt