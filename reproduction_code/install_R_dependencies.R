args <- commandArgs(TRUE)
lib <- if (length(args)) args[1] else 'Rlib'
dir.create(lib,recursive=TRUE,showWarnings=FALSE)
install.packages(c('metafor','metadat','mathjaxr','pbapply','numDeriv','digest'),repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/',lib=lib)
# This installs mirror-current packages. Reproduction requires checking exact
# versions against R_sessionInfo.txt before accepting the environment.
