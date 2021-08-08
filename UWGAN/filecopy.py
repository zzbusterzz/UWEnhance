import os
import shutil

pathofgenres = './results_color1'
pathforairimages = './Air'
pathforwaterimages = './Water'

if not os.path.exists(pathforairimages):
    os.mkdir(pathforairimages)
if not os.path.exists(pathforwaterimages):
    os.mkdir(pathforwaterimages)

print("Begining copying files")
airfiles = [filename for filename in os.listdir( pathofgenres) if filename.startswith('air')]
for imgpath in airfiles:
    shutil.copy(os.path.abspath(pathofgenres+'/' + imgpath),  os.path.abspath(pathforairimages))
print("Completed copying air files")

waterfiles = [filename for filename in os.listdir( pathofgenres) if filename.startswith('fake')]
for imgpath in waterfiles:
    shutil.copy(os.path.abspath(pathofgenres+'/' + imgpath), os.path.abspath(pathforwaterimages))
print("Completed copying water files")