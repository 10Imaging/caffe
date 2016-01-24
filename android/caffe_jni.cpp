#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#define  LOG_TAG    "CAFFE_JNI"
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG, __VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

static caffe::CaffeMobile *caffe_mobile;

using std::string;
using std::vector;
using caffe::CaffeMobile;

int getTimeSec() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int)now.tv_sec;
}

string jstring2string(JNIEnv *env, jstring jstr) {
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string str(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  return str;
}

JNIEXPORT void JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_setNumThreads(
    JNIEnv *env, jobject thiz, jint numThreads) {
  int num_threads = numThreads;
#ifdef USE_EIGEN
  omp_set_num_threads(num_threads);
#else
  openblas_set_num_threads(num_threads);
#endif
}

JNIEXPORT void JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_enableLog(JNIEnv *env,
                                                         jobject thiz,
                                                         jboolean enabled) {}

JNIEXPORT jint JNICALL Java_com_tenimaging_android_caffe_CaffeMobile_loadModel(
    JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
  CaffeMobile::Get(jstring2string(env, modelPath),
                   jstring2string(env, weightsPath));
  return 0;
}

JNIEXPORT void JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_setMeanWithMeanFile(
    JNIEnv *env, jobject thiz, jstring meanFile) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetMean(jstring2string(env, meanFile));
}

JNIEXPORT void JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_setMeanWithMeanValues(
    JNIEnv *env, jobject thiz, jfloatArray meanValues) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  int num_channels = env->GetArrayLength(meanValues);
  jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
  vector<float> mean_values(ptr, ptr + num_channels);
  caffe_mobile->SetMean(mean_values);
}

JNIEXPORT void JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_setScale(JNIEnv *env,
                                                        jobject thiz,
                                                        jfloat scale) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetScale(scale);
}

jint JNIEXPORT JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_predictImagePath(JNIEnv* env, jobject thiz, jstring imgPath)
{
    const char *img_path = env->GetStringUTFChars(imgPath, 0);
    caffe::vector<caffe::caffe_result> top_k = caffe_mobile->predict_top_k(string(img_path), 3);
    LOGD("top-1 result: %d %f", top_k[0].synset,top_k[0].prob);
        
    env->ReleaseStringUTFChars(imgPath, img_path);
    //TODO return probability
    return top_k[0].synset;
}

jint JNIEXPORT JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_predictImage(JNIEnv* env, jobject thiz, jlong cvmat_img, jint numResults, jintArray synsetList, jfloatArray probList)
{
    cv::Mat& cv_img = *(cv::Mat*)(cvmat_img);
    caffe::vector<caffe::caffe_result> top_k = caffe_mobile->predict_top_k(cv_img, numResults);
    LOGD("top-1 result: %d %f", top_k[0].synset,top_k[0].prob);

    jint *c_synsetList;
    c_synsetList = (env)->GetIntArrayElements(synsetList,NULL);
    jfloat *c_probList;
    c_probList = (env)->GetFloatArrayElements(probList,NULL);
    
    if (c_synsetList == NULL || c_probList == NULL){
        LOGE("Error getting array");
        return -1;
    }
    
    for (int i=0; i<numResults; i++)
    {
        c_synsetList[i] = top_k[i].synset;
        c_probList[i]   = top_k[i].prob;
    }
    
    // release the memory so java can have it again
    (env)->ReleaseIntArrayElements(synsetList, c_synsetList,0);
    (env)->ReleaseFloatArrayElements(probList, c_probList,0);

    return top_k[0].synset;
}

JNIEXPORT jobjectArray JNICALL
Java_com_tenimaging_android_caffe_CaffeMobile_extractFeatures(
    JNIEnv *env, jobject thiz, jstring imgPath, jstring blobNames) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<vector<float>> features = caffe_mobile->ExtractFeatures(
      jstring2string(env, imgPath), jstring2string(env, blobNames));

  jobjectArray array2D =
      env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
  for (size_t i = 0; i < features.size(); ++i) {
    jfloatArray array1D = env->NewFloatArray(features[i].size());
    if (array1D == NULL) {
      return NULL; /* out of memory error thrown */
    }
    // move from the temp structure to the java structure
    env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
    env->SetObjectArrayElement(array2D, i, array1D);
  }
  return array2D;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
