set VAR_1=https://pngimage.net/wp-content/uploads/2019/05/tree-png-architecture-2.png
c:
call "C:\Users\Florin Jurchis\Anaconda3\Scripts\"activate base
python "C:\Users\Florin Jurchis\Desktop\Training\JAVA\cvgt-py-side\opencv-semantic-segmentation\segment.py" %VAR_1%
conda deactivate
exit /b 0