package com.mediatek.gldelegatebench;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.design.widget.BottomNavigationView;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.NumberPicker;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Button;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

public class MainActivity extends AppCompatActivity {

    private TextView mTextMessage;
    private TextView resultMessage;
    private NumberPicker numberPicker;
    private NNModel mModel;
    private boolean enableGPU = false;
    private boolean enableNNAPI = false;
    private int numberOfThreads = 1;

    private MappedByteBuffer modelBuffer;
    private Interpreter.Options options;
    private Interpreter interpreter;

    private int warmupRuns = 5;
    private int loops = 50;

    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private BottomNavigationView.OnNavigationItemSelectedListener mOnNavigationItemSelectedListener
            = new BottomNavigationView.OnNavigationItemSelectedListener() {

        @Override
        public boolean onNavigationItemSelected(@NonNull MenuItem item) {
            switch (item.getItemId()) {
                case R.id.navigation_image_classfication:
                    mTextMessage.setText(R.string.title_image_classification);
                    if (mModel != null)
                        mModel = null;
                    {
                        int iShape[] = {1 * 224 * 224 * 3 * 4};
                        int oShape[] = {1 * 1001 * 4};
                        mModel = new NNModel("mobilenet_v1_1.0_224.tflite", iShape, oShape);
                        try {
                            modelBuffer = loadModelFile("mobilenet_v1_1.0_224.tflite");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return true;
                case R.id.navigation_pose_estimation:
                    mTextMessage.setText(R.string.title_pose_estimation);
                    if (mModel != null)
                        mModel = null;
                    {
                        // input: 1x353x257x3
                        // output:  1x23x17x17,  1x23x17x34, 1x23x17x64,  1x23x17x1
                        int iShape[] = {1 * 353 * 257 * 3 * 4};
                        int oShape[] = {1 * 23 * 17 * 17 * 4, 1 * 23 * 17 * 34 * 4, 1 * 23 * 17 * 64 * 4, 1 * 23 * 17 * 1 * 4};
                        mModel = new NNModel("multi_person_mobilenet_v1_075_float.tflite", iShape, oShape);
                        try {
                            modelBuffer = loadModelFile("multi_person_mobilenet_v1_075_float.tflite");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return true;
                case R.id.navigation_segmentation:
                    mTextMessage.setText(R.string.title_segmentation);
                    if (mModel != null)
                        mModel = null;
                    {
                        int iShape[] = {1 * 257* 257 * 3 * 4};
                        int oShape[] = {1 * 257* 257 * 21 * 4};
                        mModel = new NNModel("deeplabv3_257_mv_gpu.tflite", iShape, oShape);
                        try {
                            modelBuffer = loadModelFile("deeplabv3_257_mv_gpu.tflite");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return true;
                case R.id.navigation_object_detection:
                    mTextMessage.setText(R.string.title_object_detection);
                    // mobile_ssd_v2_float_coco.tflite
                    if (mModel != null)
                        mModel = null;
                    {
                        // input: 1x320x320x3
                        // output:  1x23x17x17,  1x23x17x34, 1x23x17x64,  1x23x17x1
                        int iShape[] = {1 * 320 * 320 * 3 * 4};
                        int oShape[] = {1 * 2034 * 4 * 4, 1 * 2034 * 91 * 4};
                        mModel = new NNModel("mobile_ssd_v2_float_coco.tflite", iShape, oShape);
                        try {
                            modelBuffer = loadModelFile("mobile_ssd_v2_float_coco.tflite");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return true;
                case R.id.navigation_contours:
                    mTextMessage.setText(R.string.title_contours);
                    if (mModel != null)
                        mModel = null;
                    {
                        // input: 1x192x192x3
                        // output: 1x1x1x266, 1x1x1x1x4
                        int iShape[] = {1 * 192 * 192 * 3 * 4};
                        int oShape[] = {1 * 1 * 1 * 266 * 4, 1 * 1 * 1 * 1 * 4};
                        mModel = new NNModel("contours.tflite", iShape, oShape);
                        try {
                            modelBuffer = loadModelFile("contours.tflite");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return true;
            }
            return false;
        }
    };

    public void onRadioButtonClicked(View view) {
        boolean checked = ((RadioButton) view).isChecked();

        switch(view.getId()) {
            case R.id.cpuButton:
                if (checked) {
                    enableGPU = false;
                    enableNNAPI = false;
                }
                break;
            case R.id.gpuButton:
                if (checked) {
                    enableGPU = true;
                    enableNNAPI = false;
                }
                break;
            case R.id.nnapiButton:
                if (checked) {
                    enableNNAPI = true;
                    enableGPU = false;
                }
                break;

        }
    }

    private OnClickListener mOnButtonClickListener = new OnClickListener() {

        private Object[] allocateInputBuffers(int[] shapes){
            int i_size = shapes.length;
            Object inputs[] = new Object[i_size];
            for (int i=0; i < i_size; i++) {
                ByteBuffer i_bytes = ByteBuffer.allocate(shapes[i]);
                inputs[i] = i_bytes;
            }
            return inputs;
        }

        private Map<Integer, Object> allocateOutputBuffers(int[] shapes){
            int o_size = shapes.length;
            Map<Integer, Object> outputs = new HashMap<>();
            for (int i=0; i < o_size; i++) {
                ByteBuffer o_bytes = ByteBuffer.allocate(shapes[i]);
                outputs.put(i, o_bytes);
            }
            return outputs;
        }

        public void onClick(View view) {
            long startTime, stopTime, accTime = 0;

            options = new Interpreter.Options();
            options.setNumThreads(numberOfThreads);
            GpuDelegate delegate = new GpuDelegate();
            if (enableGPU)
                options.addDelegate(delegate);
            options.setUseNNAPI(enableNNAPI);
            options.setAllowFp16PrecisionForFp32(true);

            interpreter = new Interpreter(modelBuffer, options);

            for (int i=0; i < warmupRuns; i++) {
                Object inputs[] = allocateInputBuffers(mModel.getInputShapes());
                Map<Integer, Object> outputs = allocateOutputBuffers(mModel.getOutputShapes());

                interpreter.runForMultipleInputsOutputs(inputs, outputs);
            }

            for (int i=0; i < loops; i++) {
                Object inputs[] = allocateInputBuffers(mModel.getInputShapes());
                Map<Integer, Object> outputs = allocateOutputBuffers(mModel.getOutputShapes());

                startTime = System.currentTimeMillis();
                interpreter.runForMultipleInputsOutputs(inputs, outputs);
                stopTime = System.currentTimeMillis();
                accTime += (stopTime - startTime);
            }
            Log.i("here: ", "time: " + accTime/loops);
            resultMessage.setText("avg time: " + accTime/loops + " ms");
        }
}   ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTextMessage = (TextView) findViewById(R.id.message);
        resultMessage = (TextView) findViewById(R.id.textView);
        resultMessage.setText("Results:");
        BottomNavigationView navigation = (BottomNavigationView) findViewById(R.id.navigation);
        navigation.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener);

        numberPicker = (NumberPicker) findViewById(R.id.numberPicker);
        numberPicker.setMinValue(1);
        numberPicker.setMaxValue(10);
        numberPicker.setValue(1);
        numberPicker.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                numberOfThreads = newVal;
            }
        });

        Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(mOnButtonClickListener);
        Log.i("here", "here");
        int iShape[] = {1*224*224*3*4};
        int oShape[] = {1*1001*4};
        mModel = new NNModel("mobilenet_v1_1.0_224.tflite", iShape, oShape);
        try {
            modelBuffer = loadModelFile(mModel.getModelName());
        } catch (Exception e) {
        }
    }
}
