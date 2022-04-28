library(fslr)
library(RNifti)
library(abind)
library(stringr)
library(extrantsr)
options(fsl.path ='/usr/local/fsl')
options(fsl.outputtype = 'NIFTI_GZ')

#set the path to project location (Windows)
setwd('/mnt/d')

#Get the template
aal.template = readNIfTI('Template_not_skull_stripped.nii')
aal.atlas = readNIfTI('AAL_space-MNI152NLin6_res-2x2x2.nii')
##get file names of images
cur_path = getwd()
image_path = paste0(cur_path, '/all_tau/ADNI')
file_names = dir(image_path)
image_names = list()
for(i in file_names){
  cpath = paste0(image_path, '/', i)
  cpath = paste0(cpath, '/', dir(cpath))
  sub_files = dir(cpath)
  sub_names = c()
  for(j in 1:length(sub_files)){
    spath = paste0(cpath, '/', sub_files[j])
    spath = paste0(spath, '/', dir(spath))
    spath = paste0(spath, '/', dir(spath))
    sub_names = c(sub_names, spath)
  }
  image_names[[i]] = sub_names
}


#register images and save them
for(name in names(image_names)){
  new_folder = paste0('RegisteredTAU/', name)
  dir.create(new_folder)
  files = image_names[[name]]
  num.files = length(files)
  for(i in 1:num.files){
    cur.img = readNIfTI(files[i])
    out.file = paste0(new_folder, '/aal', i, '.nii')
    #brain extraction
    cur.img = fslbet(cur.img)
    #image registration
    aal.reg = ants_regwrite(filename = cur.img, template.file = aal.template, 
                            typeofTransform = 'Affine', outfile = out.file)
  }
}

tmp = readNIfTI("RegisteredTAU/002_S_0413/aal1.nii")
