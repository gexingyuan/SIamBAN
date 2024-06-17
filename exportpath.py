import argparse
import sys
from pathlib import Path
 
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
parser.add_argument("--deletepath", type=str, default="")
args = parser.parse_args()
 
# sys.path.append(r'newlib_path')
 
if __name__ == '__main__':
 
    exportpath = eval(repr(args.path).replace('\\', '\\\\'))
    exportpathlist=exportpath.split('AND')
    # print(pathlist)
    # print(sys.executable)
 
    deletepath = eval(repr(args.deletepath).replace('\\', '\\\\'))
    deletepathlist = deletepath.split('AND')
 
    filepath=sys.prefix+'\Lib\site-packages\myExportPath.pth'
    file = Path(filepath)
 
    if not file.exists():
        # print("File doesn't exist, this code will first create one and then add the paths!!!")
        with file.open('w') as f:
            for path in exportpathlist:
                f.write(path)
                f.write("\n")
        existingPaths = file.read_text()
        existingPathslist = filter(None, existingPaths.split('\n'))
        with file.open('w') as f:  # 创建并写入。
            for existingPath in existingPathslist:
                deleteFlag=False
                for path in deletepathlist:
                    if existingPath == path:
                        deleteFlag=True
                        break
                if deleteFlag==False:
                    f.write(existingPath)
                    f.write("\n")
        existingPaths = file.read_text()
        print("The export paths are: \n"+existingPaths)
    else:
        # print("The file exists, this code will add the paths to the existing file!!!")
        existingPaths=file.read_text()
        existingPathslist = filter(None,existingPaths.split('\n'))
        with file.open('a') as f:
            for path in exportpathlist:
                existingFlag=False
                for existingPath in existingPathslist:
                    if existingPath == path:
                        existingFlag=True
                        print("warning: the path "+path+" already exists")
                        break
                if existingFlag==False:
                    f.write(path)
                    f.write("\n")
        existingPaths = file.read_text()
        existingPathslist = filter(None, existingPaths.split('\n'))
        with file.open('w') as f:
            for existingPath in existingPathslist:
                deleteFlag = False
                for path in deletepathlist:
                    if existingPath == path:
                        deleteFlag = True
                        break
                if deleteFlag == False:
                    f.write(existingPath)
                    f.write("\n")
        existingPaths = file.read_text()
        print("The export paths are: \n" + existingPaths)
    # print(sys.path)