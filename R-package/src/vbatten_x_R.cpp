#include <Rcpp.h>
#include <cstring>
#include <vector>
#include <stdexcept>

// C ABI declarations
extern "C" {
void*       vbx_learner_create(const char*);
int         vbx_set_data(void*, const float*, const float*, long, long);
int         vbx_train(void*, int);
int         vbx_predict(void*, const float*, long, long, float*);
int         vbx_save(void*, const char*);
int         vbx_load(void*, const char*);
float       vbx_train_loss(void*);
int         vbx_num_stages(void*);
void        vbx_destroy(void*);
const char* vbx_last_error();
}

static void CheckStatus(int status) {
    if (status != 0) Rcpp::stop(vbx_last_error());
}

// [[Rcpp::export]]
SEXP vbx_r_train(Rcpp::NumericVector X_flat,
                 Rcpp::NumericVector y,
                 int nrows, int ncols,
                 std::string params_json,
                 int nrounds) {
    void* h = vbx_learner_create(params_json.c_str());
    if (!h) Rcpp::stop(vbx_last_error());

    std::vector<float> Xf(X_flat.begin(), X_flat.end());
    std::vector<float> yf(y.begin(), y.end());
    CheckStatus(vbx_set_data(h, Xf.data(), yf.data(), nrows, ncols));
    CheckStatus(vbx_train(h, nrounds));

    return Rcpp::XPtr<void>(h, true, [](void* p) { vbx_destroy(p); });
}

// [[Rcpp::export]]
Rcpp::NumericVector vbx_r_predict(SEXP handle_sexp,
                                   Rcpp::NumericVector X_flat,
                                   int nrows, int ncols) {
    void* h = Rcpp::XPtr<void>(handle_sexp).get();
    std::vector<float> Xf(X_flat.begin(), X_flat.end());
    std::vector<float> out(nrows);
    CheckStatus(vbx_predict(h, Xf.data(), nrows, ncols, out.data()));
    return Rcpp::NumericVector(out.begin(), out.end());
}

// [[Rcpp::export]]
void vbx_r_save(SEXP handle_sexp, std::string path) {
    void* h = Rcpp::XPtr<void>(handle_sexp).get();
    CheckStatus(vbx_save(h, path.c_str()));
}

// [[Rcpp::export]]
SEXP vbx_r_load(std::string path) {
    void* h = vbx_learner_create("{}");
    if (!h) Rcpp::stop(vbx_last_error());
    CheckStatus(vbx_load(h, path.c_str()));
    return Rcpp::XPtr<void>(h, true, [](void* p) { vbx_destroy(p); });
}
