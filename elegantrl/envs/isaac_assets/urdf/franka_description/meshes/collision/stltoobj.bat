REM SET PATH=%PATH%;C:/Tools/Assimp/bin/x64/
REM forfiles /m *.dae /c "cmd /c assimp export @file @fname.obj --verbose --show-log -ptv"

SET PATH=%PATH%;C:/Program Files/VCG/MeshLab/
forfiles /m *.stl /c "cmd /c meshlabserver -i @file -o @fname.obj -m vn -s stltoobj.mlx"