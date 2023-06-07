package com.example.antrecognition

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.antrecognition.ml.Detect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.Float.max
import java.lang.Float.min
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity(), View.OnClickListener {

    val paint = Paint()
    lateinit var selectBtn: Button
    lateinit var imageView: ImageView
    lateinit var textView: TextView
    private lateinit var imgSampleOne: ImageView
    private lateinit var imgSampleTwo: ImageView
    private lateinit var imgSampleThree: ImageView

    lateinit var bitmap: Bitmap
    lateinit var model: Detect
    lateinit var labels: List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        labels = FileUtil.loadLabels(this, "labelmap.txt")
        model = Detect.newInstance(this)

        paint.color = Color.BLUE
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5.0f

        Log.d("labels", labels.toString())

        selectBtn = findViewById(R.id.selectBtn)
        imageView = findViewById(R.id.imageView)
        textView = findViewById(R.id.textView)
        imgSampleOne = findViewById(R.id.imgSampleOne)
        imgSampleTwo = findViewById(R.id.imgSampleTwo)
        imgSampleThree = findViewById(R.id.imgSampleThree)


        var intent = Intent()
        intent.action = Intent.ACTION_GET_CONTENT
        intent.type = "image/*"

        selectBtn.setOnClickListener {
            startActivityForResult(intent, 101)
        }
        imgSampleOne.setOnClickListener(this)
        imgSampleTwo.setOnClickListener(this)
        imgSampleThree.setOnClickListener(this)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 101) {
            var uri = data?.data;
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            val inputBuffer = prepareInputImage(bitmap)
            get_predictions(inputBuffer);
        }
        if(requestCode == 100) {
            bitmap =  BitmapFactory.decodeResource(resources, R.drawable.sampleone, BitmapFactory.Options().apply {
                inMutable = true
            })
            val inputBuffer = prepareInputImage(bitmap)
            get_predictions(inputBuffer);
        }
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.imgSampleOne -> {
                bitmap =  BitmapFactory.decodeResource(resources, R.drawable.sampleone, BitmapFactory.Options().apply {
                    inMutable = true
                })
                val inputBuffer = prepareInputImage(bitmap)
                get_predictions(inputBuffer);
            }
            R.id.imgSampleTwo -> {
                bitmap =  BitmapFactory.decodeResource(resources, R.drawable.sampletwo, BitmapFactory.Options().apply {
                    inMutable = true
                })
                val inputBuffer = prepareInputImage(bitmap)
                get_predictions(inputBuffer);
            }
            R.id.imgSampleThree -> {
                bitmap =  BitmapFactory.decodeResource(resources, R.drawable.samplethree, BitmapFactory.Options().apply {
                    inMutable = true
                })
                val inputBuffer = prepareInputImage(bitmap)
                get_predictions(inputBuffer);
            }
        }
    }

    private fun prepareInputImage(bitmap: Bitmap): ByteBuffer {

        val bitmap = Bitmap.createScaledBitmap(bitmap, 320, 320, true)
        val input = ByteBuffer.allocateDirect(320*320*3*4).order(ByteOrder.nativeOrder())
        for (y in 0 until 320) {
            for (x in 0 until 320) {
                val px = bitmap.getPixel(x, y)

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = (r - 127.5F) / 127.5F
                val gf = (g - 127.5F) / 127.5F
                val bf = (b - 127.5F) / 127.5F

                input.putFloat(rf)
                input.putFloat(gf)
                input.putFloat(bf)
            }
        }
        return input;
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    private fun get_predictions(inputBuffer: ByteBuffer) {
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val height = tensorImage.height
        val width = tensorImage.width

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(inputBuffer)

        val outputs = model.process(inputFeature0)
        val scores = outputs.outputFeature0AsTensorBuffer.floatArray
        val locations = outputs.outputFeature1AsTensorBuffer.floatArray
        val numOfPredictions = outputs.outputFeature2AsTensorBuffer.floatArray
        val classes = outputs.outputFeature3AsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        paint.textSize = height/20f
        paint.strokeWidth = height/150f

        var count = 0
        scores.forEachIndexed { index, fl ->
            if(fl > 0.6){
                count++
                var x = index
                x *= 4

                var xmin = max(1F, locations.get(x+1)*width)
                var ymin = max(1F, locations.get(x)*height)
                var xmax = min(width.toFloat(), locations.get(x+3)*width)
                var ymax = min(height.toFloat(), locations.get(x+2)*height)

                paint.color = colors.get(index)
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(xmin, ymin, xmax, ymax), paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(fl.toString(), xmin, ymin, paint)
            }

            imageView.setImageBitmap(mutable)
        }

        val textV = "Num of predictions: $count"
        textView.setText(textV)
    }
}
