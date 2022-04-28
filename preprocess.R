library(dplyr)
#assign DXCHANGE for ADNI1
setwd('Y:/PhD/Fall 2020/RA')
dxsum = read.csv("Diagnosis/DXSUM_PDXCONV_ADNIALL.csv")
attach(dxsum)
dxsum$DXCHANGE[DXCONV==0 & DXCURREN==1] = 1
dxsum$DXCHANGE[DXCONV==0 & DXCURREN==2] = 2
dxsum$DXCHANGE[DXCONV==0 & DXCURREN==3] = 3
dxsum$DXCHANGE[DXCONV==1 & DXCONTYP==1] = 4
dxsum$DXCHANGE[DXCONV==1 & DXCONTYP==3] = 5
dxsum$DXCHANGE[DXCONV==1 & DXCONTYP==2] = 6
dxsum$DXCHANGE[DXCONV==2 & DXREV==1] = 7
dxsum$DXCHANGE[DXCONV==2 & DXREV==2] = 8
dxsum$DXCHANGE[DXCONV==2 & DXREV==3] = 9
detach(dxsum)

#assign baseline diagnosis, including EMCI and SMC
arm <- read.csv("Enrollment/ARM.csv")
armVars <- c("RID","Phase","ARM","ENROLLED")
dxsumVars <- c("RID","Phase","VISCODE", "VISCODE2",
               "DXCHANGE","DXDDUE", 'DXDSEV')
dxarm <- merge(subset(dxsum, select=dxsumVars), subset(arm,
                                                       select=armVars), by=c("RID", "Phase"))
baseData <- dxarm[dxarm$VISCODE2=='bl' & dxarm$ENROLLED
                  %in% c(1,2,3),]

attach(baseData)
baseData$baselineDx[(DXCHANGE %in% c(1,7,9)) & ARM != 11 ] = 1
baseData$baselineDx[(DXCHANGE %in% c(1,7,9)) & ARM == 11 ] = 2
baseData$baselineDx[(DXCHANGE %in% c(2,4,8)) & ARM == 10 ] = 3
baseData$baselineDx[(DXCHANGE %in% c(2,4,8)) & ARM != 10 ] = 4
baseData$baselineDx[(DXCHANGE %in% c(3,5,6))] = 5
detach(baseData)

baseVars <- c("RID","baselineDx")
dxarm <- merge( dxarm, subset(baseData, select=baseVars),
                by=c("RID"))

#keep EXAMDATE
registry <- read.csv("Enrollment/REGISTRY.csv")
regVars <-c("RID", "Phase", "VISCODE", "VISCODE2",
            "EXAMDATE", "PTSTATUS", "RGCONDCT", "RGSTATUS",
            "VISTYPE")
dxarm_reg <- merge(dxarm, subset(registry, select=regVars),
                   by=c("RID", "Phase", "VISCODE"))%>%arrange(RID)
dxarm_reg = dxarm_reg%>%arrange(RID, EXAMDATE)
dxarm_reg$DXCURREN[dxarm_reg$DXCHANGE %in% c(1,7,9)] = 0
dxarm_reg$DXCURREN[dxarm_reg$DXCHANGE %in% c(2,4,8)] = 1
dxarm_reg$DXCURREN[dxarm_reg$DXCHANGE %in% c(3,5,6)] = 2
dxarm_reg$VISCODE2 = dxarm_reg$VISCODE2.x
dxarm.sum = dxarm_reg%>%group_by(RID)%>%
  filter(n_distinct(ARM)>1)

cate1 = dxarm_reg%>%mutate(DXCURREN = ifelse(DXCURREN == 0, 0, ifelse(DXCURREN == 1, 1, ifelse(DXCURREN == 2 & DXDSEV == 1, 2, NA))))

cate2 =  dxarm_reg%>%filter(DXCURREN == 0 | DXCURREN == 1 | DXDDUE == 2)%>%
  mutate(DXCURREN = ifelse(DXCURREN == 0, 0, ifelse(DXCURREN == 1, 1, 2)))

#normal
t1 = dxarm_reg%>%filter(DXCURREN == 0)
#mci
t2 = dxarm_reg%>%filter(DXCURREN == 1)
#mild dementia (mild Alzheimer's disease)
t3 = dxarm_reg%>%filter(DXDSEV == 1 & DXCURREN == 2) 
#dementia due to other etiology (small sample size, not use)
t4 = dxarm_reg%>%filter(DXDDUE == 2)
