package com.example.tutorialse;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.os.Environment;
import android.view.MenuItem;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.Queue;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private Mat mRgba, mGray;
    //private JavaCameraView mOpenCvCameraView;
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "OCVSample::Activity";

    private static final Scalar FACE_RECT_COLOR = new Scalar(255, 0, 255);
    private static final Scalar ROI_RECT_COLOR = new Scalar(0, 255, 255);

    private TextView mScoreText;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    private Mat mCurrentRGBA;
    private Mat mCurrentGray;
    private Mat ROI;
    private Mat face;
    //    private MediaRecorder mRecorder;
    private int FrameCounter = 10;
    private VideoWriter mVideoWriter;
    private ImageView mVideoCameraButton;
    private String mVideoFileName;
    private boolean mIsRecording;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;
    private Rect mROIRect;
    private Queue<Mat> mROIFrameBuffer;
    private Mat mROIToStore;
    private MatOfRect LastCaptureOfImages;
    private Rect FirstFace;
    // private Rect LastFace;

    BaseLoaderCallback mCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // initialize opencv variables on manager!
                    mCurrentRGBA = new Mat();
                    mCurrentGray = new Mat();
                    mIsRecording = false;
//        mRecorder = new MediaRecorder();
                    mVideoWriter = new VideoWriter();
                    mROIRect = new Rect();
                    mROIFrameBuffer = new LinkedList<Mat>();
                    mROIToStore = null;

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mCallBack);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.MyView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGray = new Mat();
        face = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        FrameCounter ++;
        mCurrentRGBA = inputFrame.rgba(); //Stores the Input Frame
        mCurrentGray = inputFrame.gray(); //Stores the Input Frame in Gray Color

        Imgproc.cvtColor(mCurrentRGBA, mCurrentGray, Imgproc.COLOR_RGBA2GRAY);

        if (mAbsoluteFaceSize == 0) {
            int height = mCurrentGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        //At the last frames we use the method rect to detect and square the frames
        MatOfRect faces = new MatOfRect(); // Creates a class name Faces typed as MatOfRect

        if (mJavaDetector != null){
            mJavaDetector.detectMultiScale(mCurrentGray, faces, 1.1, 2, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            Log.i(TAG, "Detecting Face"+faces);
        }
        Rect[] facesArray = faces.toArray(); // Obtain the face array captured in the frames
        //Mat ROI = new Mat();
        // mROIToStore = forehead(faces);

        //   Log.i(TAG,"onCameraFrame: >>>>>>>>> Frame Counter___"+FrameCounter);
        //   Log.i(TAG, "OnCameraFrame: >>>>>>>>> Face Array Length___"+facesArray.length);
        Log.i(TAG,"onCameraFrame: >>>>>>>>> Frame Counter___"+FrameCounter);

        //if(FrameCounter>=1000){
        //    FrameCounter = 0;
        //}

        // Proximo Passo seria Tratar isso como um buffer
        if(facesArray.length>=2) {
            Log.i(TAG, "OnCameraFrame: >>>>>>>> FacesArrayLength: " + facesArray.length);
            Log.i(TAG,"onCameraFrame: >>>>>>>> Frame Counter: " + FrameCounter);
            LastCaptureOfImages = faces;
            FirstFace = LastCaptureOfImages.toArray()[0];
            Rect LastFace = facesArray[1];

            if (!FirstFace.empty()) {
                Log.i(TAG, "OnCameraFrame: >>>>>>>> There is something at the First Face: "+FirstFace);

            }
            if(!LastFace.empty()){
                Log.i(TAG, "OnCameraFrame: >>>>>>>> There is something at the Last Face: "+LastFace);
            }

            FrameCounter = 0;

            if(FrameCounter == 0){
                Log.i(TAG, "OnCameraFrame: >>>>>>>> FrameCounter has been reinitialized!!");
            }

            // for (int i = 0; i < ROIArray.length; i++){
            //     Imgproc.rectangle(mCurrentRGBA, ROIArray[i].tl(), ROIArray[i].br(), ROI_RECT_COLOR, 3);
            // }
        }

        //Draw the face on original frame RGBA
        // if(facesArray.length < 2){
        //     TakePictureFrameRGB(mCurrentRGBA); // Call the Function to take Separated Pictures
        // }
        for (int i = 0; i < facesArray.length; i++){
            Imgproc.rectangle(mCurrentRGBA, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }

        return mCurrentRGBA;
    }

    public Mat forehead(Mat face){                                                              //esta função separa a testa no rosto detectado
        //a
        return face.submat(new Rect(45, 0, 210, 90));
    }

    public void TakePictureFrameRGB(Mat mCurrentRGBA) {
        // Add permission to store the images in local storage
        // Now we create a new mat that you want to save
        Mat SaveMat = new Mat();
        // Convert image from RGBA to BGRA
        Imgproc.cvtColor(mCurrentRGBA, SaveMat, Imgproc.COLOR_RGB2BGR);
        //Now create a new folder to Save the Pictures
        File folder = new File(Environment.getExternalStorageDirectory().getPath() + "/FaceDetector");
        // if the folder don't be created
        boolean success = true;
        if (!folder.exists()) {
            success = folder.mkdirs();
        }
        // now we create a unique file name for that image that was taken
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateAndTime = sdf.format(new Date());
        String fileName = Environment.getExternalStorageDirectory().getPath() + "/FaceDetector" + currentDateAndTime + ".jpg";
        //write save_mat
        Imgcodecs.imwrite(fileName, SaveMat);

    }
}