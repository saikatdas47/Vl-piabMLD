#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(nlme))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1) stop("Usage: run_secondary_mixed_effects.R PACKAGE_DIR")
package_dir <- normalizePath(args[[1]])
out_dir <- file.path(package_dir, "secondary_analysis")
d <- read.csv(file.path(out_dir, "h3_h6_cell_level_analysis_data.csv"), stringsAsFactors=FALSE)
d$dataset_id <- factor(d$dataset_id)
d$condition <- relevel(factor(d$condition), ref="C2")
d$model <- relevel(factor(d$model), ref="logistic_regression")

full_formula <- delta_AUROC ~ condition * (
  z_log_sample_size + z_log_feature_to_sample_ratio +
  z_minority_prevalence + z_missingness_rate + model
)
control <- lmeControl(opt="optim", maxIter=500, msMaxIter=500, niterEM=100, returnObject=TRUE)

fit_model <- function(formula, weights=NULL) {
  lme(formula, random=~1|dataset_id, data=d, method="ML", weights=weights,
      na.action=na.fail, control=control)
}

full <- fit_model(full_formula)
reduced_formulas <- list(
  H3=update(full_formula, . ~ . - z_log_sample_size - condition:z_log_sample_size),
  H4=update(full_formula, . ~ . - z_log_feature_to_sample_ratio - condition:z_log_feature_to_sample_ratio),
  H5=update(full_formula, . ~ . - z_minority_prevalence - condition:z_minority_prevalence),
  H6=update(full_formula, . ~ . - model - condition:model)
)

lrt_rows <- list()
for (hyp in names(reduced_formulas)) {
  reduced <- fit_model(reduced_formulas[[hyp]])
  cmp <- anova(reduced, full)
  lrt_rows[[hyp]] <- data.frame(
    hypothesis=hyp,
    df_difference=cmp$df[2]-cmp$df[1],
    likelihood_ratio=cmp$L.Ratio[2],
    p_value=cmp$`p-value`[2],
    stringsAsFactors=FALSE
  )
}
lrt <- do.call(rbind, lrt_rows)
write.csv(lrt, file.path(out_dir, "h3_h6_likelihood_ratio_tests.csv"), row.names=FALSE)

coef_table <- as.data.frame(summary(full)$tTable)
coef_table$term <- rownames(coef_table)
rownames(coef_table) <- NULL
names(coef_table)[1:5] <- c("estimate","standard_error","df","t_value","p_value")
coef_table$ci95_low <- coef_table$estimate - 1.96*coef_table$standard_error
coef_table$ci95_high <- coef_table$estimate + 1.96*coef_table$standard_error
coef_table <- coef_table[,c("term","estimate","standard_error","df","t_value","p_value","ci95_low","ci95_high")]
write.csv(coef_table, file.path(out_dir, "mixed_effects_coefficients.csv"), row.names=FALSE)

# Condition-specific moderator slopes as linear combinations of fixed effects.
b <- fixef(full); V <- vcov(full)
contrast <- function(base, condition) {
  L <- rep(0, length(b)); names(L) <- names(b)
  if (base %in% names(L)) L[base] <- 1
  term <- paste0("condition",condition,":",base)
  reverse <- paste0(base,":condition",condition)
  if (condition != "C2" && term %in% names(L)) L[term] <- 1
  if (condition != "C2" && reverse %in% names(L)) L[reverse] <- 1
  est <- sum(L*b); se <- sqrt(as.numeric(t(L)%*%V%*%L))
  data.frame(condition=condition, estimate=est, standard_error=se,
             ci95_low=est-1.96*se, ci95_high=est+1.96*se)
}
slopes <- list()
for (hyp in c("H3","H4","H5")) {
  base <- switch(hyp, H3="z_log_sample_size", H4="z_log_feature_to_sample_ratio", H5="z_minority_prevalence")
  block <- do.call(rbind, lapply(c("C2","C3","C4","C5"), function(cc) contrast(base,cc)))
  block$hypothesis <- hyp
  block$moderator <- base
  slopes[[hyp]] <- block
}
write.csv(do.call(rbind,slopes), file.path(out_dir,"condition_specific_moderator_slopes.csv"), row.names=FALSE)

# Model-family contrasts at mean continuous covariates, by condition.
newdata <- expand.grid(
  condition=levels(d$condition), model=levels(d$model),
  z_log_sample_size=0, z_log_feature_to_sample_ratio=0,
  z_minority_prevalence=0, z_missingness_rate=0
)
X <- model.matrix(delete.response(terms(full_formula)), newdata)
X <- X[,names(b),drop=FALSE]
pred <- as.numeric(X%*%b)
pred_se <- sqrt(diag(X%*%V%*%t(X)))
newdata$estimate <- pred; newdata$standard_error <- pred_se
newdata$ci95_low <- pred-1.96*pred_se; newdata$ci95_high <- pred+1.96*pred_se
write.csv(newdata, file.path(out_dir,"model_family_adjusted_estimates.csv"), row.names=FALSE)

# Diagnostics.
res <- resid(full, type="normalized")
fitted_values <- fitted(full)
shapiro <- shapiro.test(res)
hetero <- try(fit_model(full_formula, weights=varIdent(form=~1|condition)), silent=TRUE)
if (!inherits(hetero,"try-error")) {
  hetero_cmp <- anova(full, hetero)
  hetero_lrt <- hetero_cmp$L.Ratio[2]
  hetero_p <- hetero_cmp$`p-value`[2]
  hetero_aic_delta <- AIC(hetero)-AIC(full)
} else {
  hetero_lrt <- NA; hetero_p <- NA; hetero_aic_delta <- NA
}
Xfixed <- model.matrix(full_formula, d)
condition_number <- kappa(Xfixed, exact=TRUE)
numeric_corr <- cor(d[,c("z_log_sample_size","z_log_feature_to_sample_ratio","z_minority_prevalence","z_missingness_rate")])
write.csv(numeric_corr, file.path(out_dir,"moderator_correlation_matrix.csv"))
write.csv(data.frame(fitted=fitted_values,normalized_residual=res,dataset_id=d$dataset_id,
                     condition=d$condition,model=d$model),
          file.path(out_dir,"mixed_effects_residuals.csv"),row.names=FALSE)

# Leave-one-dataset-out stability of the main fixed-effect vector.
loo <- list()
for (id in levels(d$dataset_id)) {
  dd <- droplevels(d[d$dataset_id != id,])
  fit <- try(lme(full_formula, random=~1|dataset_id, data=dd, method="ML",
                 na.action=na.fail, control=control), silent=TRUE)
  if (inherits(fit,"try-error")) {
    loo[[id]] <- data.frame(omitted_dataset=id,converged=FALSE,max_abs_coefficient_shift=NA)
  } else {
    common <- intersect(names(b),names(fixef(fit)))
    shift <- max(abs(fixef(fit)[common]-b[common]))
    loo[[id]] <- data.frame(omitted_dataset=id,converged=TRUE,max_abs_coefficient_shift=shift)
  }
}
loo <- do.call(rbind,loo)
write.csv(loo,file.path(out_dir,"leave_one_dataset_out_diagnostics.csv"),row.names=FALSE)

diagnostics <- data.frame(
  diagnostic=c("model_converged","observations","dataset_clusters","fixed_effect_parameters",
               "condition_number","residual_shapiro_W","residual_shapiro_p",
               "heteroscedasticity_lrt","heteroscedasticity_p","heterogeneous_minus_homogeneous_AIC",
               "leave_one_dataset_out_failures","maximum_leave_one_out_coefficient_shift"),
  value=c(TRUE,nrow(d),nlevels(d$dataset_id),length(b),condition_number,unname(shapiro$statistic),shapiro$p.value,
          hetero_lrt,hetero_p,hetero_aic_delta,sum(!loo$converged),max(loo$max_abs_coefficient_shift,na.rm=TRUE))
)
write.csv(diagnostics,file.path(out_dir,"mixed_effects_diagnostics.csv"),row.names=FALSE)

saveRDS(full,file.path(out_dir,"mixed_effects_model.rds"))
capture.output(summary(full),file=file.path(out_dir,"mixed_effects_model_summary.txt"))
cat("Mixed-effects analysis complete\n")
