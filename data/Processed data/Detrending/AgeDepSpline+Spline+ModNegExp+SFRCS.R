library(dplR)
library(tools)
dir.create("D:/Detrending4", showWarnings = FALSE, recursive = TRUE)
dir.create("D:/mistake", showWarnings = FALSE, recursive = TRUE)
rwl_files <- list.files("D:/Tree rings/rwl/width", pattern = "\\.rwl$", full.names = TRUE)
for (file_path in rwl_files) {
  tryCatch({
    file_name <- basename(file_path)
    csv_name <- file_path_sans_ext(file_name)
    lines <- readLines(file_path, warn = FALSE)
    if (length(lines) <= 3) stop("Insufficient data after removing the first three lines")
    temp_file <- tempfile(fileext = ".rwl")
    writeLines(lines[-c(1:3)], temp_file)
    rwl_data <- read.rwl(temp_file)
    unlink(temp_file)
    if (nrow(rwl_data) < 2 || ncol(rwl_data) < 1) stop("Invalid RWL data")
    rwi_negexp <- detrend(rwl_data, method = "ModNegExp")
    rwi_spline <- detrend(rwl_data, method = "Spline", nyrs = 30)
    rwi_agedep <- detrend(rwl_data, method = "AgeDepSpline")
    crn_negexp <- chron(rwi_negexp, prefix = "")
    crn_spline <- chron(rwi_spline, prefix = "")
    crn_agedep <- chron(rwi_agedep, prefix = "")
    if (!"std" %in% names(crn_negexp) || !"std" %in% names(crn_spline) || !"std" %in% names(crn_agedep)) stop("Chronology construction failed")
    df_negexp <- data.frame(year = as.numeric(rownames(crn_negexp)), ModNegExp = crn_negexp$std, samp.depth = crn_negexp$samp.depth)
    df_spline <- data.frame(year = as.numeric(rownames(crn_spline)), Spline = crn_spline$std)
    df_agedep <- data.frame(year = as.numeric(rownames(crn_agedep)), AgeDepSpline = crn_agedep$std)
    rw <- as.matrix(rwl_data)
    storage.mode(rw) <- "double"
    ages <- matrix(NA_integer_, nrow = nrow(rw), ncol = ncol(rw), dimnames = dimnames(rw))
    for (j in seq_len(ncol(rw))) {
      idx <- which(is.finite(rw[, j]))
      if (length(idx) > 0) ages[idx, j] <- seq_along(idx)
    }
    max_age <- suppressWarnings(max(ages, na.rm = TRUE))
    if (!is.finite(max_age) || max_age < 1) stop("Unable to calculate cambial ages")
    rwi_rcs_init <- detrend(rwl_data, method = "RCS")
    crn_rcs_init <- chron(rwi_rcs_init, prefix = "")
    make_chron_vec <- function(ch) {
      yrs_all <- as.numeric(rownames(rwl_data))
      y <- as.numeric(rownames(ch))
      v <- ch$std
      out <- rep(NA_real_, length(yrs_all))
      names(out) <- yrs_all
      ii <- match(y, yrs_all)
      ok <- !is.na(ii)
      out[ii[ok]] <- v[ok]
      out
    }
    chron_prev <- make_chron_vec(crn_rcs_init)
    max_iter <- 8
    tol <- 1e-4
    crn_sfrcs_now <- NULL
    for (it in seq_len(max_iter)) {
      Cmat <- matrix(chron_prev, nrow = nrow(rw), ncol = ncol(rw), byrow = FALSE)
      sf <- matrix(NA_real_, nrow = nrow(rw), ncol = ncol(rw), dimnames = dimnames(rw))
      ok <- is.finite(rw) & is.finite(Cmat) & Cmat > 0
      sf[ok] <- rw[ok] / Cmat[ok]
      rc_sf <- rep(NA_real_, max_age)
      for (a in seq_len(max_age)) {
        vals <- sf[ages == a]
        vals <- vals[is.finite(vals)]
        if (length(vals) > 0) rc_sf[a] <- mean(vals)
      }
      rc_sf[!is.finite(rc_sf) | rc_sf <= 0] <- NA_real_
      rwi_sfrcs <- matrix(NA_real_, nrow = nrow(rw), ncol = ncol(rw), dimnames = dimnames(rw))
      for (j in seq_len(ncol(rw))) {
        aj <- ages[, j]
        ok2 <- is.finite(rw[, j]) & !is.na(aj)
        idx <- which(ok2)
        if (length(idx) > 0) {
          curve_vals <- rc_sf[aj[idx]]
          good <- is.finite(curve_vals) & curve_vals > 0
          if (any(good)) rwi_sfrcs[idx[good], j] <- rw[idx[good], j] / curve_vals[good]
        }
      }
      rwi_sfrcs_df <- as.data.frame(rwi_sfrcs)
      rownames(rwi_sfrcs_df) <- rownames(rwl_data)
      colnames(rwi_sfrcs_df) <- colnames(rwl_data)
      crn_sfrcs_now <- chron(rwi_sfrcs_df, prefix = "")
      chron_now <- make_chron_vec(crn_sfrcs_now)
      common <- is.finite(chron_now) & is.finite(chron_prev)
      if (!any(common)) break
      diffv <- max(abs(chron_now[common] - chron_prev[common]), na.rm = TRUE)
      chron_prev <- chron_now
      if (!is.finite(diffv) || diffv < tol) break
    }
    if (is.null(crn_sfrcs_now)) stop("SFRCS chronology construction failed")
    crn_sfrcs <- crn_sfrcs_now
    df_sfrcs <- data.frame(year = as.numeric(rownames(crn_sfrcs)), SFRCS = crn_sfrcs$std)
    merged_crn <- Reduce(function(x, y) merge(x, y, by = "year", all = TRUE), list(df_negexp, df_spline, df_agedep, df_sfrcs))
    rwi_mat <- as.matrix(rwi_agedep)
    rwi_years <- as.numeric(rownames(rwi_agedep))
    eps_vec <- rep(NA_real_, nrow(rwi_mat))
    for (i in seq_along(rwi_years)) {
      valid_samples <- which(is.finite(rwi_mat[i, ]))
      n <- length(valid_samples)
      if (n >= 2) {
        mat <- rwi_mat[, valid_samples, drop = FALSE]
        cor_mat <- suppressWarnings(cor(mat, use = "pairwise.complete.obs"))
        r_vals <- cor_mat[lower.tri(cor_mat)]
        r_vals <- r_vals[is.finite(r_vals)]
        if (length(r_vals) > 0) {
          rbar <- mean(r_vals)
          den <- n * rbar + 1 - rbar
          if (is.finite(den) && den != 0) eps_vec[i] <- n * rbar / den
        }
      }
    }
    eps_df <- data.frame(year = rwi_years, EPS = eps_vec)
    merged_crn <- merge(merged_crn, eps_df, by = "year", all = TRUE)
    merged_crn <- merged_crn[order(merged_crn$year), ]
    write.csv(merged_crn, file = paste0("D:/Detrending4/", csv_name, ".csv"), row.names = FALSE)
    message("SUCCESS: ", file_name)
  }, error = function(e) {
    file.copy(file_path, paste0("D:/mistake/", basename(file_path)), overwrite = TRUE)
    message("FAILED: ", basename(file_path), " | ", e$message)
  })
}