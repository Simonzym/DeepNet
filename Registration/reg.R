library(fslr)
library(RNifti)
library(abind)
library(stringr)
library(extrantsr)
options(fsl.path ='/usr/local/fsl')
options(fsl.outputtype = 'NIFTI_GZ')

#set the path to project location (Windows)
setwd('/mnt/y/PhD/Fall 2020/RA')

#Get the template
aal.template = readNIfTI('AAL_space-MNI152NLin6_res-2x2x2.nii')

# #get the PET image
# img = readNIfTI('ADNI_003_S_1074_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20070312161623427_1_S27694_I44225.nii')
# img_bet = fslbet(img)
# img_bias = fsl_biascorrect(img)
# orthographic(img)
# orthographic(img_bet)
# 
# img.bet = fslbet(img, retimg = T)
# orthographic(img.bet)
# 
# #affine transformation
# img.desikan = flirt(reffile = desikan.template,
#             infile = img,
#             outfile = 'Reg/to_desikan', dof = 12)
# img.aal = flirt(infile = img, reffile = aal.template,
#                   outfile = 'Reg/to_aicha', dof = 12)
# double_ortho(img.desikan, desikan.template)
# double_ortho(img.aal, aal.template)
# 
# #read some pets
# #follow up
# t1 = readNIfTI('ttt/ADNI_003_S_1059_PT_Talairach_Warped_Br_20081217150252101_1_S33888_I131218.nii')
# t2 = readNIfTI('ttt/ADNI_003_S_1059_PT_Coreg,_warp,_norm_Br_20100929131716683_56_S33888_I193939.nii')
# t3 = readNIfTI('ttt/ADNI_003_S_1059_MR_Tx_Origin,_Aligned_Baseline,_Spatially_Normalized_Br_20110112101730038_S33888_I210463.nii')
# t4 = readNIfTI('ttt/ADNI_003_S_1059_MR_Tx_Origin,_Aligned_Baseline,_Spatially_Normalized,_Smoothed_Br_20080313163609500_S33888_I97405.nii')
# #baseline
# t5 = readNIfTI('ADNI_003_S_1059_MR_Tx_Origin,_Spatially_Normalized_Br_20110112094645691_S25144_I210346.nii')
# t6 = readNIfTI('ADNI_003_S_1059_MR_Tx_Origin,_Spatially_Normalized,_Smoothed_Br_20080314123329226_S25144_I97683.nii')

##get file names of images
cur_path = getwd()
image_path = paste0(cur_path, '/Data_Download/FDGDIF12')
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
  new_folder = paste0('Data/RegisteredDIF12/', name)
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

#left and right angular gyrus
LA = aal.template==69
RA = aal.template==70
#left and right posterior cingulate gyrus
LP = aal.template==39
RP = aal.template==40
BP = LP+RP
#left and right inferior temporal gyrus
LT = aal.template==93
RT = aal.template==94

visual.roi = function(template, mask, ax = c(45, 50, 65)){

  template_mask = template
  template_mask[!mask] = NA
  orthographic(template, template_mask, xyz = ax)

  }
#function that the index of ROIs given ROI mask
extract.roi = function(mask, cut = TRUE){

  x.mask = apply(mask, 1, function(x) any(x==1))
  y.mask = apply(mask, 2, function(x) any(x==1))
  z.mask = apply(mask, 3, function(x) any(x==1))
  
  x.index = which(x.mask==1)
  y.index = which(y.mask==1)
  z.index = which(z.mask==1)
  
  if(cut){
    x.index = min(x.index):max(x.index)
    y.index = min(y.index):max(y.index)
    z.index = min(z.index):max(z.index)
  }
  
  dimension = c(length(x.index), length(y.index), length(z.index))
  return(list(x = x.index, y = y.index, z = z.index, d = dimension))
}
#index for each ROI
roi = c('LA', 'RA', 'LP', 'RP', 'BP', 'LT', 'RT')
for(name in roi){
  assign(paste(name,'.ind', sep = ''), extract.roi(get(name)))
}
#check the dimension of each ROI (cubic area)
for(name in roi){
  print(name)
  tmp = get(paste(name,'.ind', sep = ''))
  print(tmp$d)
}

#get file names of registered images
cur_path = getwd()
image_path = paste0(cur_path, '/Data/RegisteredDIF12')
file_names = dir(image_path)
image_names = list()
for(i in file_names){
  cpath = paste0(image_path, '/', i)
  sub_files = dir(cpath)
  sub_names = c()
  for(j in 1:length(sub_files)){
    spath = paste0(cpath, '/', sub_files[j])
    sub_names = c(sub_names, spath)
  }
  image_names[[i]] = sub_names
}

#extract BP/BC and LT from the registered images and save them
for(name in names(image_names)){
  bp.folder = paste0('Data/ROIDIF12/BP/', name)
  lt.folder = paste0('Data/ROIDIF12/LT/', name)
  dir.create(bp.folder)
  dir.create(lt.folder)
  files = image_names[[name]]
  num.files = length(files)
  for(i in 1:num.files){
    cur.img = readNIfTI(files[i])
    #extract BP/BC
    img.bp = cur.img
    #setting other area to 0
    img.bp[!BP] = 0
    img.bp = img.bp[BP.ind$x, BP.ind$y, BP.ind$z]
    #extract LT
    img.lt = cur.img
    #setting other area to 0
    img.lt[!LT] = 0
    img.lt = img.lt[LT.ind$x, LT.ind$y, LT.ind$z]
    #save them
    img.bp = as.nifti(img.bp, value = cur.img)
    img.lt = as.nifti(img.lt, value = cur.img)
    bp.file = paste0(bp.folder, '/extract_', i)
    lt.file = paste0(lt.folder, '/extract_', i)
    writeNIfTI(img.bp, bp.file)
    writeNIfTI(img.lt, lt.file)
  }
}



