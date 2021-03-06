library(dplyr)
library(oro.nifti)
demo = read.csv('Demographics/PTDEMOG.csv')
#tmp = readNIfTI('Data/RegisteredDIF12/003_S_1057/aal1')
#FDG PET

##1.obtain all the patients through ADNI2 and have 5 or more PET images available
fdg = read.csv('PET info/FDG/BAIPETNMRCFDG_11_22_19.csv')

#save the month as time for further selection
fdg.info = fdg%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  group_by(RID)%>%
  mutate(index = row_number())%>%
  ungroup(RID)

#choose only the ones having at least three visits (including outcome)
id_3 = fdg.info%>%count(RID)%>%filter(n>2)

#abandon
# fdg.diff = fdg.info%>%filter(RID%in%id_3$RID)%>%group_by(RID)%>%arrange(time)%>%
#    mutate(diff = time - lag(time, default = NA))%>%
#    mutate(diff2 = time - lag(time, n = 2, default = NA))%>%
#    mutate(diff3 = time - lag(time, n = 3, default = NA))%>%
#    mutate(diff4 = time - lag(time, n = 4, default = NA))%>%
#    ungroup(RID)%>%
#    arrange(RID, time)

#join image dataset and cognitive score/status dataset
fdg_adas = fdg.info%>%
  full_join(diagnosis, by = c('RID', 'VISCODE2', 'time'))%>%
  filter(!is.na(DXCURREN))%>%
  filter(RID %in% id_3$RID)%>%
  group_by(RID)%>%arrange(time)%>%
  #time difference of the target and the last image
  mutate(diff = time - lag(time, default = NA))%>%
  mutate(diff2 = time - lag(time, n = 2, default = NA))%>%
  mutate(diff3 = time - lag(time, n = 3, default = NA))%>%
  mutate(diff4 = time - lag(time, n = 4, default = NA))%>%
  mutate(is.image = !is.na(MODALITY))%>%
  mutate(positionRID = row_number())%>%
  #calculate images available for building models
  mutate(num.image1 = cumsum(is.image) - is.image)%>%
  mutate(num.image2 = num.image1 - lag(is.image, default=NA))%>%
  mutate(num.image3 = num.image2 - lag(is.image, n = 2, default=NA))%>%
  mutate(num.image4 = num.image3 - lag(is.image, n = 3, default=NA))%>%
  ungroup(RID)

#predict on outcome 6 months later
dif6.lag1 = fdg_adas%>%group_by(RID)%>%
  filter(num.image1 >= 3 & !is.na(DXCURREN)  & diff == 6 & lag(is.image, default = NA))%>%
  #filter(row_number() == n())%>%
  ungroup(RID)

#predict on outcome 12 months later
dif12.lag1 = fdg_adas%>%group_by(RID)%>%
  filter(num.image1 >= 3 & !is.na(DXCURREN)  & diff == 12 & lag(is.image, default = NA))%>%
  #filter(row_number() == n())%>%
  ungroup(RID)

dif12.lag2 = fdg_adas%>%group_by(RID)%>%
  filter(num.image2 >= 3 & !is.na(DXCURREN)  & diff2 == 12 & lag(is.image, n=2, default = NA))%>%
  #filter(row_number() == n())%>%
  ungroup(RID)


roster = read.csv('Enrollment/ROSTER.csv')

#filter out invalid subjects, who has the same visits records several times
invalids = fdg_adas%>%count(RID, VISCODE2)%>%filter(n>1)
dif12.lag1 = filter(dif12.lag1, !RID %in% invalids$RID)
dif12.lag2 = filter(dif12.lag2, !RID %in% invalids$RID)
get_id = roster%>%select(RID, PTID)%>%filter(RID %in% c(dif12.lag1$RID, dif12.lag2$RID))%>%distinct(RID, PTID)
write.csv(paste(get_id$PTID,collapse = ','), 'FDG12_ID.csv')
#delete previous decided PTID, do not run
# abandon_id = read.csv('FDG12_ID.csv', fill = TRUE)
# abandon_ptid = unlist(strsplit(abandon_id[1,2], ','))
# paste(setdiff(get_id$PTID, abandon_ptid),collapse = ',')
# extra_id = setdiff(abandon_ptid, get_id$PTID)
# #we choose to predict on outcome 12 months later because of larger sampler size
# data_folder = paste0(getwd(), '/Data_Download/FDGDIF12/ADNI')
# 
# #delete unused subjects
# for(id in extra_id){
#   file_name = paste0(data_folder, '/', id)
#   unlink(file_name, recursive = TRUE)
# }


#graph_id
dif12.lag1$graph_id = paste0(dif12.lag1$RID, '_', dif12.lag1$time)
dif12.lag2$graph_id = paste0(dif12.lag2$RID, '_', dif12.lag2$time)
#build info for sequential images
#note each subject can have more than one graphs
#create a new variable RID.time, where time represents when the outcome was measured
#for dif12.lag1 and dif12.lag2, each row represents the outcome of a graph

#info.data contains information of sequence of images (not outcome)
#dif12 contains the outcome of each graph
info.data = data.frame()
dif12 = rbind(dif12.lag1, dif12.lag2)

for(row in 1:dim(dif12.lag1)[1]){
  outcome.data = dif12.lag1[row,]
  one.data = fdg_adas%>%filter(RID == outcome.data$RID & positionRID < outcome.data$positionRID & is.image)%>%
    mutate(graph_id = paste0(RID, '_', outcome.data$time))
  info.data = rbind(info.data, one.data)
}


for(row in 1:dim(dif12.lag2)[1]){
  outcome.data = dif12.lag2[row,]
  one.data = fdg_adas%>%filter(RID == outcome.data$RID & positionRID < (outcome.data$positionRID - 1) & is.image)%>%
    mutate(graph_id = paste0(RID, '_', outcome.data$time))
  info.data = rbind(info.data, one.data)
}


#get info of mmse and cdr as well
info.data = info.data%>%left_join(all.mmse.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'), keep = FALSE)
info.data = info.data%>%left_join(all.cdr.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'), keep = FALSE)
info.data = info.data%>%left_join(all.adas.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'), keep = FALSE)

dif12 = dif12%>%left_join(all.mmse.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'))
dif12 = dif12%>%left_join(all.cdr.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'))
dif12 = dif12%>%left_join(all.adas.info, by = c('RID', 'VISCODE2' = 'VISCODE2.x'))

#impute cognitive score/cognitive status in sequence
while(anyNA(info.data$MMSCORE) & anyNA(info.data$TOTAL11) & anyNA(info.data$CDGLOBAL)){
  
  info.data = info.data%>%group_by(graph_id)%>%
    mutate(CDGLOBAL = ifelse(is.na(CDGLOBAL), lag(CDGLOBAL, default = NA), CDGLOBAL))%>%
    mutate(MMSCORE = ifelse(is.na(MMSCORE), lag(MMSCORE, default = NA), MMSCORE))%>%
    mutate(TOTAL11 = ifelse(is.na(TOTAL11), lag(TOTAL11, default = NA), TOTAL11))%>%
    mutate(DXNA = is.na(DXCURREN))
}



info.last = info.data%>%group_by(graph_id)%>%filter(row_number() == n())%>%ungroup(graph_id)
info.last%>%count(RID)%>%count(n)

#impute ADAS by the last visit if NA in dif12


info.visits = info.last%>%group_by(RID)%>%filter(positionRID == max(positionRID))%>%ungroup(RID)
info.allvis = info.data%>%filter(graph_id %in% info.visits$graph_id)

info.all.vis = data.frame()
for(rid in unique(dif12$RID)){
  last.visit = dif12%>%filter(RID == rid)%>%group_by(RID)%>%filter(row_number() == n())
  tmp = info.allvis%>%filter(RID == rid)%>%mutate(lastdx = last.visit$DXCURREN)
  info.all.vis = rbind(info.all.vis, tmp)
}
library(ggplot2)
ggplot(info.all.vis)+geom_point(aes(x = time/12, y = factor(RID), color = factor(lastdx)), size = 0.1, shape = 20)+ scale_y_discrete(breaks = NULL) + ylab('')+
  xlab('Time from baseline (in years)') + ggtitle('ADNI Studies by Subject')+
  theme_classic()+ scale_color_discrete(name = 'Group', labels = c('Normal', 'MCI', 'AD'))+
  theme(plot.title = element_text(hjust = 0.5, size = 14), panel.border = element_rect(colour = "black", fill=NA, size=0.5))

#demogrphics
base.data = fdg.data%>%filter(RID %in% info.allvis$RID)%>%filter(VISCODE2 == 'bl')%>%
  select(age, RID, PTGENDER, PTEDUCAT, PTHAND, PTNOTRT)
basescore = info.all.vis%>%filter(VISCODE2 == 'bl')
basedata = data.frame()
for(rid in unique(dif12$RID)){
  last.visit = dif12%>%filter(RID == rid)%>%group_by(RID)%>%filter(row_number() == n())
  tmp = base.data%>%filter(RID == rid)%>%mutate(lastdx = last.visit$DXCURREN)
  basedata = rbind(basedata, tmp)
}
#get demographics
data.use = basedata%>%filter(lastdx == 1)
score.use = basescore%>%filter(lastdx == 1)
mean(data.use$age)
sd(data.use$age)

mean(data.use$PTGENDER==1)

mean(score.use$TOTAL11)
sd(score.use$TOTAL11)

mean(score.use$MMSCORE)
sd(score.use$MMSCORE)

mean(score.use$CDGLOBAL)
sd(score.use$CDGLOBAL)
#number of visits
vissum = info.all.vis%>%filter(lastdx == 2)%>%group_by(RID)%>%summarise(count = n())
mean(vissum$count)
sd(vissum$count)

#5-fold cv for subjects (not for graphs)
all.rid = unique(info.data$RID)
set.seed(100)
folds = sample(1:5, length(all.rid), replace = TRUE)
cv.folder = 'Y:/PhD/Fall 2020/RA/Code/Info/graphDIF12'
for(i in 1:5){
  train.rid = all.rid[folds != i]
  test.rid = all.rid[folds == i]
  
  train.gid = unique(info.data$graph_id[info.data$RID %in% train.rid])
  test.gid = unique(info.data$graph_id[info.data$RID %in% test.rid])
  
  train.file = paste0(cv.folder, '/cv', i, '/train.csv')
  test.file = paste0(cv.folder, '/cv', i, '/test.csv')
  
  write.csv(train.gid, train.file)
  write.csv(test.gid, test.file)
}
tmp = dif12%>%filter(RID %in% all.rid[folds == 5])
table(tmp$DXCURREN)
#Them create information for CNN (use dif12 only)
roi.path = function(roi){
  #get the file_names (PIDs)
  cur_path = getwd()
  #should change the folder name according to your need
  folder_name = '/Data/ROIDIF12/'
  image_path = paste0(cur_path, folder_name, roi)
  file_names = dir(image_path)
  image_names = list()
  #for each subject
  for(i in file_names){
    cpath = paste0(image_path, '/', i)
    sub_files = dir(cpath)
    sub_names = c()
    #save all the images
    for(j in 1:length(sub_files)){
      spath = paste0(cpath, '/', sub_files[j])
      sub_names = c(sub_names, spath)
    }
    image_names[[i]] = sub_names
  }
  return(image_names)
}
bp_names = roi.path('BP')
lt_names = roi.path('LT')
la_names = roi.path('LA')
ra_names = roi.path('RA')
rt_names = roi.path('RT')
pet_names = roi.path('PET')
#list all the visit by graph_id
id.vis = list()
for(gid in dif12$graph_id){
  id.vis[[gid]] = info.data%>%arrange(RID, time)%>%filter(graph_id == gid)%>%select(VISCODE2, RID, graph_id, DXNA,
                                                                            TOTAL11, DXCURREN, time, MMSCORE, CDGLOBAL, index)
}

#save the names by rid
save.by.gid = function(image_names, roi){
  
  for(gid in names(id.vis)){
    visits = id.vis[[gid]]
    ptid = filter(get_id, RID == sub('_.*', '', gid))
    images = image_names[[ptid$PTID]]
    img.index = visits$index
    #should change saving path
    write.csv(data.frame(images = images[img.index], visits),
              paste('Code/Info/DIF12/', roi, '/', gid, '.csv', sep = ''),quote = FALSE)
  }
  
}
save.by.gid(bp_names, 'BP')
save.by.gid(lt_names, 'LT')
save.by.gid(la_names, 'LA')
save.by.gid(ra_names, 'RA')
save.by.gid(rt_names, 'RT')
save.by.gid(pet_names, 'PET')
#get sequence of baseline information of each subject (age, gender)
fdg.baseline = registry%>%filter(RID %in% dif12$RID & VISCODE2 == 'bl')%>%
  dplyr::select(RID, VISCODE2, EXAMDATE)%>%inner_join(demo.1, by = 'RID')%>%
  mutate(age = year(EXAMDATE) - PTDOBYY)

#outcome and baseline
fdg.base.out = dif12%>%inner_join(fdg.baseline, by = c('RID'))%>%
  inner_join(get_id, by = 'RID')%>%
  select(graph_id, RID, PTGENDER, age, TOTAL11, PTID, DXCURREN, VISCODE.x, time, MMSCORE, CDGLOBAL)
write.csv(fdg.base.out, paste('Code/Info/DIF12/base_out.csv'))


tmp = read.csv('Code/Info/DIF12/base_out.csv')
tmp%>%count(RID)%>%count(n)
#initiate dataframe for edges and graphs
all.edges = data.frame()
all.graphs = data.frame()
for (gid in names(id.vis)){
  info.gid = id.vis[[gid]]
  #create edge information
  ##combination of row number
  num_nodes = nrow(info.gid)
  src.tar = combn(1:num_nodes, 2)
  times = info.gid$time
  ##weights equal to difference of time
  weights = 1/(times[src.tar[2,]] - times[src.tar[1,]])
  edges = data.frame(graph_id = gid, src = src.tar[1,],
                     dst = src.tar[2,], weight = weights)
  
  #create graph information
  item = filter(dif12, graph_id == gid)
  item.last = filter(info.last, graph_id == gid)
  graphs = data.frame(graph_id = gid, label = item$DXCURREN, 
                      num_nodes = num_nodes, ADAS = ifelse(is.na(item$TOTAL11), item.last$TOTAL11, item$TOTAL11),
                      MMSE = item$MMSCORE, CDR = item$CDGLOBAL)
  
  #combine information
  all.graphs = rbind(all.graphs, graphs)
  all.edges = rbind(all.edges, edges)
}

cur.path = getwd()
save.folder = paste0(cur.path, '/Code/Info/graphDIF12')
write.csv(all.graphs, paste0(save.folder, '/graphs.csv'), quote = FALSE)
write.csv(all.edges, paste0(save.folder, '/edges.csv'), quote = FALSE)











#only consider the image data set (abandon)
# dif6_rid = fdg.diff%>%group_by(RID)%>%mutate(positionRID = 1:n())%>%
#   arrange(RID)%>%
#   filter(row_number() >= 4)%>%
#   filter((diff == 6 & )%>%
#   filter(row_number() == n())
# 
# #303 unique RIDs
# dif12_rid = fdg.diff%>%group_by(RID)%>%mutate(positionRID = 1:n())%>%
#   arrange(RID)%>%
#   filter((row_number() >= 4 & diff == 12) | (row_number() >= 5 & diff2 == 12))
# adas.last = dif12_rid%>%inner_join(all.info, by = c('RID', 'time'))%>%
#   filter(!is.na(DXCURREN) & TOTAL11 != -1)
# 
# dif18_rid = fdg.diff%>%group_by(RID)%>%mutate(positionRID = 1:n())%>%
#   arrange(RID)%>%
#   filter(row_number() >= 4)%>%
#   filter(diff == 18 | diff2 == 18 | diff3 == 18)%>%
#   filter(row_number() == n())
# 
# dif24_rid = fdg.diff%>%group_by(RID)%>%mutate(positionRID = 1:n())%>%
#   arrange(RID)%>%
#   filter(row_number() >= 4)%>%
#   filter(diff == 24 | diff2 == 24 | diff3 == 24 | diff4 == 24)%>%
#   filter(row_number() == n())
# 
# 
# setdiff(dif12_rid$RID, dif24_rid$RID)
# setdiff(dif24_rid$RID, dif12_rid$RID)


# #select those whose 5th visit cognitive score is available

# #exclude 1205 and 1157, both have missing TOTAL11; 1205 has missing
# #DXCURREN (1157 has it)
# 
# setdiff(dif12_rid$RID, adas.last$RID)
# 
# roster = read.csv('Enrollment/ROSTER.csv')
# get_id = roster%>%select(RID, PTID)%>%filter(RID %in% dif12_rid$RID)%>%distinct(RID, PTID)
# write.csv(paste(get_id$PTID,collapse = ','), 'FDG12_ID.csv')

