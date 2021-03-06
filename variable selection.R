library(tidyr)
library(stringr)
library(lme4)
library(lubridate)
library(dplyr)
demo = read.csv('Demographics/PTDEMOG.csv')
#longitudinal analysis of cognitive scores on biomarkers(fdg)
##1. scores and biomarkers
adas1 = read.csv('Scores/ADAS/ADASSCORES.csv')
adas23 = read.csv('Scores/ADAS/ADAS_ADNIGO23.csv')
cdr = read.csv('Scores/CDR/CDR.csv')
mmse = read.csv('Scores/Other Scores/MMSE.csv')
fdg.roi = read.csv('PET info/FDG/UCBERKELEYFDG_05_28_20.csv')
vis2 = read.csv('Enrollment/ADNI2_VISITID.csv')
adas1$Phase = 'ADNI1'

adas1.use = adas1%>%select(Phase, RID, VISCODE, TOTAL11,
                                            TOTAL13 = TOTALMOD)%>%mutate(VISCODE2 = VISCODE)
adas23.use = adas23%>%select(Phase, RID, VISCODE2, VISCODE,
                                TOTAL11=TOTSCORE, TOTAL13)

adas.use = rbind(adas1.use, adas23.use)%>%drop_na(TOTAL11)%>%arrange(RID)
adas.use$TOTAL11[adas.use$TOTAL11==-4] = -1

adas.fdg = adas.use%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  drop_na(time)%>%
  arrange(RID, time)

mmse.use = mmse%>%select(Phase, RID, VISCODE, VISCODE2, MMSCORE)%>%
  mutate(VISCODE = ifelse(VISCODE == 'sc', 'bl', VISCODE))%>%
  mutate(VISCODE2 = ifelse(VISCODE2 == 'sc', 'bl', VISCODE2))%>%
  filter(VISCODE2 != '' & VISCODE2 != 'f' & VISCODE2 != 'uns1')%>%
  drop_na(MMSCORE)

cdr.use = cdr%>%select(Phase, RID, VISCODE, VISCODE2, CDGLOBAL)%>%
  mutate(VISCODE = ifelse(VISCODE == 'sc', 'bl', VISCODE))%>%
  mutate(VISCODE2 = ifelse(VISCODE2 == 'sc', 'bl', VISCODE2))%>%
  filter(VISCODE2 != '' & VISCODE2 != 'f' & VISCODE2 != 'uns1')%>%
  drop_na(CDGLOBAL)

mmse.fdg = mmse.use%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  drop_na(time)%>%
  arrange(RID, time)

cdr.fdg = cdr.use%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  drop_na(time)%>%
  arrange(RID, time)

cdr.fdg$CDGLOBAL[cdr.fdg$CDGLOBAL == -1] = NA
mmse.fdg$MMSCORE[mmse.fdg$MMSCORE == -1] = NA
adas.fdg$TOTAL11[adas.fdg$TOTAL11 == -1] = NA

diagnosis = dxarm_reg%>%filter(VISCODE2 != 'sc' & VISCODE2 != 'uns1')%>%
  select(RID, Phase, VISCODE, DXCURREN, VISCODE2)%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  drop_na(time)%>%
  arrange(RID, time)

#data set includes both cognitive score and cognitive status
all.adas.info = adas.fdg%>%inner_join(diagnosis, by = c('RID', 'VISCODE'), keep = FALSE)%>%select(RID, VISCODE2.x, TOTAL11)
all.mmse.info = mmse.fdg%>%inner_join(diagnosis, by = c('RID', 'VISCODE'), keep = FALSE)%>%select(RID, VISCODE2.x, MMSCORE)
all.cdr.info = cdr.fdg%>%inner_join(diagnosis, by = c('RID', 'VISCODE'), keep = FALSE)%>%select(RID, VISCODE2.x, CDGLOBAL)



##2. choose those selected patients and relevant variables
fdg.roi.tmp = fdg.roi%>%select(-update_stamp)%>%
  mutate(region = paste0(ROILAT,ROINAME,sep=' '))%>%select(RID, VISCODE, VISCODE2, EXAMDATE, MEAN, region)

fdg.mean = fdg.roi.tmp%>%tidyr::spread(region, MEAN)%>%arrange(RID)
colnames(fdg.mean)[5:9] = c('BC', 'LA', 'LT', 'RA', 'RT')

demo.1 = demo%>%arrange(RID)%>%
  select(RID, PTGENDER, PTDOBYY, PTHAND, PTMARRY, PTEDUCAT, PTWORKHS,
         PTNOTRT, PTHOME, PTPLANG, PTETHCAT, PTRACCAT, USERDATE)%>%group_by(RID)%>%dplyr::slice(1)

fdg.data = fdg.mean%>%full_join(all.adas.info, by = c('RID', 'VISCODE2'='VISCODE2.x'))%>%
  full_join(demo.1, by = 'RID')%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))%>%
  mutate(age = year(USERDATE) - PTDOBYY)


