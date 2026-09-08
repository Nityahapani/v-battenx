# VBattenX R Package — main API
# Requires: Rcpp, vbatten_x shared library on LD_LIBRARY_PATH

#' Train a VBattenX booster
#'
#' @param data  matrix or data.frame of features (numeric)
#' @param label numeric vector of labels
#' @param params list of training parameters
#' @param nrounds integer number of boosting rounds
#' @return vbx.Booster object
#' @export
vbx.train <- function(data, label, params = list(), nrounds = 100L) {
  if (is.data.frame(data)) data <- as.matrix(data)
  stopifnot(is.matrix(data), is.numeric(label))
  stopifnot(nrow(data) == length(label))

  params$learning_rate  <- params$learning_rate  %||% 0.1
  params$reg_lambda     <- params$reg_lambda      %||% 1.0
  params$objective      <- params$objective       %||% "regression"

  handle <- .Call("vbx_r_train",
                  as.single(data),
                  as.single(label),
                  nrow(data), ncol(data),
                  jsonlite::toJSON(params, auto_unbox = TRUE),
                  as.integer(nrounds))

  structure(list(handle = handle,
                 nrounds = nrounds,
                 params  = params),
            class = "vbx.Booster")
}

#' Predict with a VBattenX booster
#'
#' @param booster vbx.Booster object
#' @param newdata matrix of features
#' @return numeric vector of predictions
#' @export
predict.vbx.Booster <- function(booster, newdata, ...) {
  if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
  stopifnot(is.matrix(newdata))
  .Call("vbx_r_predict",
        booster$handle,
        as.single(newdata),
        nrow(newdata), ncol(newdata))
}

#' Cross-validate a VBattenX booster
#'
#' @param data  feature matrix
#' @param label label vector
#' @param params list of parameters
#' @param nfold integer number of folds
#' @param nrounds integer number of rounds
#' @return data.frame with train and test RMSE per fold
#' @export
vbx.cv <- function(data, label, params = list(), nfold = 5L, nrounds = 100L) {
  if (is.data.frame(data)) data <- as.matrix(data)
  n      <- nrow(data)
  folds  <- cut(seq_len(n), breaks = nfold, labels = FALSE)
  folds  <- sample(folds)

  results <- lapply(seq_len(nfold), function(k) {
    tr_idx  <- folds != k
    val_idx <- folds == k
    b       <- vbx.train(data[tr_idx, ], label[tr_idx], params, nrounds)
    p_tr    <- predict(b, data[tr_idx, ])
    p_val   <- predict(b, data[val_idx, ])
    list(train_rmse = sqrt(mean((p_tr  - label[tr_idx])^2)),
         test_rmse  = sqrt(mean((p_val - label[val_idx])^2)))
  })

  data.frame(
    fold        = seq_len(nfold),
    train_rmse  = sapply(results, `[[`, "train_rmse"),
    test_rmse   = sapply(results, `[[`, "test_rmse")
  )
}

#' Save a VBattenX booster to disk
#' @export
vbx.save <- function(booster, path) {
  .Call("vbx_r_save", booster$handle, as.character(path))
  invisible(booster)
}

#' Load a VBattenX booster from disk
#' @export
vbx.load <- function(path, params = list()) {
  handle <- .Call("vbx_r_load", as.character(path))
  structure(list(handle = handle, params = params), class = "vbx.Booster")
}

#' @export
print.vbx.Booster <- function(x, ...) {
  cat("VBattenX Booster\n")
  cat("  rounds:", x$nrounds, "\n")
  cat("  params:", paste(names(x$params), x$params, sep = "=", collapse = ", "), "\n")
  invisible(x)
}

#' @export
summary.vbx.Booster <- function(object, ...) {
  cat("VBattenX Booster Summary\n")
  cat("  Training rounds:", object$nrounds, "\n")
  print(object$params)
  invisible(object)
}

#' @export
plot.vbx.Booster <- function(x, ...) {
  message("Topology/residual plots available via Python: vbatten_x.field_viz")
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
