# Rscript recompute_E2.R TableS10_E2_cohort_inputs.csv meta_recomputed.csv
args <- commandArgs(TRUE)
library(metafor)
d <- read.csv(args[1]); d <- d[d$included_in_meta %in% c(TRUE,"True"),]
rows <- list()
for (key in unique(paste(d$task,d$method,d$layer,sep="|"))) {
  u <- d[paste(d$task,d$method,d$layer,sep="|")==key,]
  for (est in c("DL","REML","PM")) {
    fit <- rma.uni(yi=u$cal_int,vi=u$se^2,method=est,test="knha")
    ci <- confint(fit)
    rows[[length(rows)+1]] <- data.frame(task=u$task[1],method=u$method[1],layer=u$layer[1],estimator=est,k=nrow(u),tau=sqrt(fit$tau2),tau_lo=sqrt(ci$random[1,"ci.lb"]),tau_hi=sqrt(ci$random[1,"ci.ub"]),pooled=as.numeric(fit$b),lo=fit$ci.lb,hi=fit$ci.ub,I2=fit$I2)
  }
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
