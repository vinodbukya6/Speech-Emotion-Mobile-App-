package com.tensorflow.android.emotionclassifier

import android.Manifest
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.media.audiofx.NoiseSuppressor
import android.os.*
import android.text.TextUtils
import android.util.Log
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.karumi.dexter.Dexter
import com.karumi.dexter.MultiplePermissionsReport
import com.karumi.dexter.PermissionToken
import com.karumi.dexter.listener.PermissionRequest
import com.karumi.dexter.listener.multi.MultiplePermissionsListener
import com.ml.quaterion.noiseClassification.Recognition
import com.tensorflow.android.audio.features.*
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.math.RoundingMode
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.text.DecimalFormat
import java.util.*
import kotlin.collections.ArrayList


class MainActivity : AppCompatActivity(), MultiplePermissionsListener {

    val permissions = listOf(
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.RECORD_AUDIO
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Dexter.withActivity(this)
                .withPermissions(permissions)
                .withListener(this)
                .check()

    }

    override fun onPermissionRationaleShouldBeShown(permissions: MutableList<PermissionRequest>, token: PermissionToken) {
        // This method will be called when the user rejects a permission request
        // You must display a dialog box that explains to the user why the application needs this permission
    }

    override fun onPermissionsChecked(report: MultiplePermissionsReport) {
        // Here you have to check granted permissions
        homescreen()
    }

    fun homescreen() {
        //  val languages = resources.getStringArray(R.array.Languages)
        val externalStorage: File = Environment.getExternalStorageDirectory()

        val audioDirPath = externalStorage.absolutePath + "/audioData";

        val fileNames: MutableList<String> = ArrayList()


        File(audioDirPath).walk().forEach {

            if (it.absolutePath.endsWith(".wav")) {
                fileNames.add(it.name)
            }

        }

        // access the spinner
        val spinner = findViewById<Spinner>(R.id.spinner)

        if (spinner != null) {
            val adapter = ArrayAdapter(this,
                    android.R.layout.simple_spinner_dropdown_item, fileNames)
            spinner.adapter = adapter

        }
        val emoji = findViewById<ImageView>(R.id.emoji)
        val emotion_text = findViewById<TextView>(R.id.emotion)

        classify_button.setOnClickListener(View.OnClickListener {
            val selFilePath = spinner.selectedItem.toString()
            val audioFilePath = audioDirPath + '/' + selFilePath;

            if (!TextUtils.isEmpty(selFilePath)) {

                playAudioFile(audioFilePath)
                doInference(audioFilePath)
                //WaveSplitter.getChunks(File(audioFilePath))

            } else {
                Toast.makeText(this@MainActivity, "something went wrong!! audio path", Toast.LENGTH_LONG).show();
            }
        })


        // Restore the previous task or create a new one if necessary
        recordTask = lastCustomNonConfigurationInstance as RecordWaveTask?
        if (recordTask == null) {
            recordTask = RecordWaveTask(this)
        } else {
            recordTask!!.setContext(this)
        }

        val record_chunk = findViewById<Button>(R.id.record_chunk)
        record_chunk.setOnClickListener {
            if (record_chunk.text.toString().trim { it <= ' ' } == "stop recording") {
                if (!recordTask?.isCancelled()!! && recordTask?.getStatus() == AsyncTask.Status.RUNNING) {
                    recordTask?.cancel(false)
                } else {
                    // Toast.makeText(MainActivity.this, "Task not running.", Toast.LENGTH_SHORT).show();
                }
                record_chunk.text = "start recording"
            } else {
                record_chunk.text = "stop recording"
                launchTask()
            }
        }
    }

    fun doInference(audioFilePath: String) {

        // getDuration for audioFilePath, if duration is >  10sec split the file else don't split
        val mNumFrames1: Int
        val mSampleRate1: Int
        val mDuration1: Int
        var wavFile1: WavFile? = null
        wavFile1 = WavFile.openWavFile(File(audioFilePath))
        mNumFrames1 = wavFile1.numFrames.toInt()
        mSampleRate1 = wavFile1.sampleRate.toInt()
        mDuration1 = mNumFrames1 / mSampleRate1

        Log.i("File Duration", String.format(mDuration1.toString()))
        Log.i("File_Path", audioFilePath)
        this.to(audioFilePath)
        this.to(mDuration1)

        if (mDuration1 > 10) {
            val chunks = getSplittedChunks(File(audioFilePath)) // write splitted chunks
            //val chunks = WaveSplitter.getChunks(File(audioFilePath)) // write splitted chunks
            val result = classifyNoise(audioFilePath) // apply function
            val commaSeperatedString = result?.joinToString { it -> "\'${it.getTitle() + (String.format("(%.1f%%) ", it.getConfidence()?.times(100.0f)))}\'" }
            val topone = result?.get(0)?.getTitle()
            if (topone.equals("Anger")) {
                emoji.setImageDrawable(getDrawable(R.drawable.anger))
                emotion.setText("Anger")

            } else if (topone.equals("Fear")) {
                emoji.setImageDrawable(getDrawable(R.drawable.fear))
                emotion.setText("Fear")
            } else if (topone.equals("Joy")) {
                emoji.setImageDrawable(getDrawable(R.drawable.happy))
                emotion.setText("Joy")
            } else if (topone.equals("None")) {
                emoji.setImageDrawable(getDrawable(R.drawable.neutral))
                emotion.setText("neutral")
            } else if (topone.equals("Sad")) {
                emoji.setImageDrawable(getDrawable(R.drawable.sad))
                emotion.setText("Sad")
            }

            result_text.text = "Predicted Emotion: " + commaSeperatedString
        } else {
            val result = classifyNoise(audioFilePath) // apply function
            val commaSeperatedString = result?.joinToString { it -> "\'${it.getTitle() + (String.format("(%.1f%%) ", it.getConfidence()?.times(100.0f)))}\'" }
            val topone = result?.get(0)?.getTitle()
            if (topone.equals("Anger")) {
                emoji.setImageDrawable(getDrawable(R.drawable.anger))
                emotion.setText("Anger")

            } else if (topone.equals("Fear")) {
                emoji.setImageDrawable(getDrawable(R.drawable.fear))
                emotion.setText("Fear")
            } else if (topone.equals("Joy")) {
                emoji.setImageDrawable(getDrawable(R.drawable.happy))
                emotion.setText("Joy")
            } else if (topone.equals("None")) {
                emoji.setImageDrawable(getDrawable(R.drawable.neutral))
                emotion.setText("neutral")
            } else if (topone.equals("Sad")) {
                emoji.setImageDrawable(getDrawable(R.drawable.sad))
                emotion.setText("Sad")
            }

            result_text.text = "Predicted Emotion: " + commaSeperatedString

        }
    }

    fun classifyNoise(audioFilePath: String): ArrayList<Recognition>? {

        val mNumFrames: Int
        val mSampleRate: Int
        val mChannels: Int
        var meanMFCCValues: FloatArray = FloatArray(1)

        var predictedResult: ArrayList<Recognition>?;

        var wavFile: WavFile? = null
        try {
            wavFile = WavFile.openWavFile(File(audioFilePath))
            mNumFrames = wavFile.numFrames.toInt()
            mSampleRate = wavFile.sampleRate.toInt()
            mChannels = wavFile.numChannels
            val buffer =
                    Array(mChannels) { DoubleArray(mNumFrames) }

            var frameOffset = 0
            val loopCounter: Int = mNumFrames * mChannels / 4096 + 1
            for (i in 0 until loopCounter) {
                frameOffset = wavFile.readFrames(buffer, mNumFrames, frameOffset)
            }

            //trimming the magnitude values to 5 decimal digits
            val df = DecimalFormat("#.#####")
            df.setRoundingMode(RoundingMode.CEILING)
            val meanBuffer = DoubleArray(mNumFrames)
            for (q in 0 until mNumFrames) {
                var frameVal = 0.0
                for (p in 0 until mChannels) {
                    frameVal = frameVal + buffer[p][q]
                }
                meanBuffer[q] = df.format(frameVal / mChannels).toDouble()
            }


            //MFCC java library.
            val mfccConvert = MFCC()
            mfccConvert.setSampleRate(mSampleRate)
            val nMFCC = 40
            mfccConvert.setN_mfcc(nMFCC)
            val mfccInput = mfccConvert.process(meanBuffer)
            val nFFT = mfccInput.size / nMFCC
            val mfccValues =
                    Array(nMFCC) { DoubleArray(nFFT) }

            //loop to convert the mfcc values into multi-dimensional array
            for (i in 0 until nFFT) {
                var indexCounter = i * nMFCC
                val rowIndexValue = i % nFFT
                for (j in 0 until nMFCC) {
                    mfccValues[j][rowIndexValue] = mfccInput[indexCounter].toDouble()
                    indexCounter++
                }
            }

            //code to take the mean of mfcc values across the rows such that
            //[nMFCC x nFFT] matrix would be converted into
            //[nMFCC x 1] dimension - which would act as an input to tflite model
            meanMFCCValues = FloatArray(nMFCC)
            for (p in 0 until nMFCC) {
                var fftValAcrossRow = 0.0
                for (q in 0 until nFFT) {
                    fftValAcrossRow = fftValAcrossRow + mfccValues[p][q]
                }
                val fftMeanValAcrossRow = fftValAcrossRow / nFFT
                meanMFCCValues[p] = fftMeanValAcrossRow.toFloat()
            }

        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: WavFileException) {
            e.printStackTrace()
        }

        meanMFCCValues = meanMFCCValues
        predictedResult = loadModelAndMakePredictions(meanMFCCValues)

        return predictedResult

    }


    fun loadModelAndMakePredictions(meanMFCCValues: FloatArray): ArrayList<Recognition> {


        //load the TFLite model in 'MappedByteBuffer' format using TF Interpreter
        val tfliteModel: MappedByteBuffer =
                FileUtil.loadMappedFile(this, getModelPath())
        val tflite: Interpreter

        /** Options for configuring the Interpreter.  */
        val tfliteOptions =
                Interpreter.Options()
        tfliteOptions.setNumThreads(2)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        //obtain the input and output tensor size required by the model
        //for urban sound classification, input tensor should be of 1x40x1x1 shape
        val imageTensorIndex = 0
        val imageShape =
                tflite.getInputTensor(imageTensorIndex).shape()
        val imageDataType: DataType = tflite.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape()
        val probabilityDataType: DataType =
                tflite.getOutputTensor(probabilityTensorIndex).dataType()

        //need to transform the MFCC 1d float buffer into 1x40x1x1 dimension tensor using TensorBuffer
        val inBuffer: TensorBuffer = TensorBuffer.createDynamic(imageDataType)
        inBuffer.loadArray(meanMFCCValues, imageShape)
        val inpBuffer: ByteBuffer = inBuffer.getBuffer()
        val outputTensorBuffer: TensorBuffer =
                TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)


        //Code to transform the probability predictions into label values
        val ASSOCIATED_AXIS_LABELS = "labels.txt"
        var associatedAxisLabels: List<String>? = null
        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS)
        } catch (e: IOException) {
            Log.e("tfliteSupport", "Error reading label file", e)
        }


        val output =
                Array(1) { FloatArray(5) }
        tflite.run(inpBuffer, output)

        val testing = output;
        val t = testing


        val MAX_RESULTS: Int = 5
        val pq: PriorityQueue<Recognition> = PriorityQueue(
                MAX_RESULTS,
                Comparator<Recognition> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                    java.lang.Float.compare(rhs.getConfidence(), lhs.getConfidence())
                })


        for (i in 0 until 5) {
            pq.add(Recognition("" + i, associatedAxisLabels!![i], output[0][i]))
        }
        val results: ArrayList<Recognition> = ArrayList()
        while (!pq.isEmpty()) {
            results.add(pq.poll())
        }

        return results

    }


    fun getModelPath(): String {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "model.tflite"
    }


    // For recording the wav file and saving it to the local memory card.
    var recordTask: RecordWaveTask? = null
    fun launchTask() {
        Toast.makeText(this, "Recording started please speak...", Toast.LENGTH_SHORT).show()
        when (recordTask!!.status) {
            AsyncTask.Status.RUNNING ->               //  Toast.makeText(this, "Task already running...", Toast.LENGTH_SHORT).show();
                return
            AsyncTask.Status.FINISHED ->                 //Toast.makeText(this, "Task finished running...", Toast.LENGTH_SHORT).show();
                recordTask = RecordWaveTask(this)
            AsyncTask.Status.PENDING ->                 //Toast.makeText(this, "Task pending running...", Toast.LENGTH_SHORT).show();
                if (recordTask!!.isCancelled) {
                    recordTask = RecordWaveTask(this)
                }
        }
        val wavFile = generaFile()
        // File wavFile = new File(getFilesDir(), "recording_" + System.currentTimeMillis() / 1000 + ".wav");
        //Toast.makeText(this, wavFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        recordTask!!.execute(wavFile)
    }

    public fun generaFile(): File {
        val directory = File("/storage/emulated/0/Download/")
        if (!directory.exists()) {
            directory.mkdir()
            if (!directory.exists()) {
                Log.e(
                        "RECORD",
                        "WARNING! Directory does not exists !!!! Creation problems"
                )
            }
        }
        val filename =
                "temp.wav" //"recording_" + System.currentTimeMillis() / 1000 + ".wav";
        val newFile = File(directory, filename)
        if (!newFile.exists()) {
            try {
                newFile.createNewFile()
            } catch (e: IOException) {
                Log.e("RECORD", e.message)
            }
        }
        return newFile
    }

    override fun onRetainCustomNonConfigurationInstance(): Any? {
        recordTask!!.setContext(null)
        return recordTask
    }

    //    class RecordWaveTask(ctx: Context) :
    @SuppressLint("StaticFieldLeak")
    class RecordWaveTask(ctx: MainActivity?) :

            AsyncTask<File?, Void?, Array<Any>>() {

        //
        private val BUFFER_SIZE = 2 * AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_MASK,
                ENCODING
        )

        var ctx: MainActivity? = null
        fun setContext(ctx: MainActivity?) {
            this.ctx = ctx
        }


        override fun doInBackground(vararg files: File?): Array<Any>? {
            //protected override fun doInBackground(vararg files: File): Array<Any> {
            var audioRecord: AudioRecord? = null
            var wavOut: FileOutputStream? = null
            var startTime: Long = 0
            var endTime: Long = 0
            try {
                // Open our two resources
                audioRecord = AudioRecord(
                        AUDIO_SOURCE,
                        SAMPLE_RATE,
                        CHANNEL_MASK,
                        ENCODING,
                        BUFFER_SIZE
                )
                val ns = NoiseSuppressor.create(audioRecord.audioSessionId)
                ns.enabled = true
                wavOut = FileOutputStream(files[0])


                // Write out the wav file header
                writeWavHeader(
                        wavOut,
                        CHANNEL_MASK,
                        SAMPLE_RATE,
                        ENCODING
                )

                // Avoiding loop allocations
                val buffer = ByteArray(BUFFER_SIZE)
                var run = true
                var read: Int
                var total: Long = 0

                // Let's go
                startTime = SystemClock.elapsedRealtime()
                audioRecord.startRecording()
                while (run && !isCancelled) {
                    read = audioRecord.read(buffer, 0, buffer.size)

                    // WAVs cannot be > 4 GB due to the use of 32 bit unsigned integers.
                    if (total + read > 4294967295L) {
                        // Write as many bytes as we can before hitting the max size
                        var i = 0
                        while (i < read && total <= 4294967295L) {
                            // wavOut.write(buffer[i])

                            wavOut.write(buffer)

                            i++
                            total++
                        }
                        run = false
                    } else {
                        // Write out the entire read buffer
                        wavOut.write(buffer, 0, read)
                        total += read.toLong()
                    }
                }
            } catch (ex: IOException) {
                return arrayOf(ex)
            } finally {
                if (audioRecord != null) {
                    try {
                        if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                            audioRecord.stop()
                            endTime = SystemClock.elapsedRealtime()
                        }
                    } catch (ex: IllegalStateException) {
                        //
                    }
                    if (audioRecord.state == AudioRecord.STATE_INITIALIZED) {
                        audioRecord.release()
                    }
                }
                if (wavOut != null) {
                    try {
                        wavOut.close()
                    } catch (ex: IOException) {
                        //
                    }
                }
            }
            try {
                // This is not put in the try/catch/finally above since it needs to run
                // after we close the FileOutputStream
                files[0]?.let { updateWavHeader(it) }
            } catch (ex: IOException) {
                return arrayOf(ex)
            }
            return arrayOf(arrayOf(files[0]?.length(), endTime - startTime))
        }

        @Throws(IOException::class)
        public fun writeWavHeader(
                out: OutputStream,
                channelMask: Int,
                sampleRate: Int,
                encoding: Int
        ) {
            val channels: Short
            channels = when (channelMask) {
                AudioFormat.CHANNEL_IN_MONO -> 1
                AudioFormat.CHANNEL_IN_STEREO -> 2
                else -> throw IllegalArgumentException("Unacceptable channel mask")
            }
            val bitDepth: Short
            bitDepth = when (encoding) {
                AudioFormat.ENCODING_PCM_8BIT -> 8
                AudioFormat.ENCODING_PCM_16BIT -> 16
                AudioFormat.ENCODING_PCM_FLOAT -> 32
                else -> throw IllegalArgumentException("Unacceptable encoding")
            }
            writeWavHeader(out, channels, sampleRate, bitDepth)
        }

        @Throws(IOException::class)
        public fun writeWavHeader(
                out: OutputStream,
                channels: Short,
                sampleRate: Int,
                bitDepth: Short
        ) {
            // Convert the multi-byte integers to raw bytes in little endian format as required by the spec
            val littleBytes = ByteBuffer
                    .allocate(14)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .putShort(channels)
                    .putInt(sampleRate)
                    .putInt(sampleRate * channels * (bitDepth / 8))
                    .putShort((channels * (bitDepth / 8)).toShort())
                    .putShort(bitDepth)
                    .array()

            // Not necessarily the best, but it's very easy to visualize this way
            out.write(
                    byteArrayOf( // RIFF header
                            'R'.toByte(),
                            'I'.toByte(),
                            'F'.toByte(),
                            'F'.toByte(),  // ChunkID
                            0,
                            0,
                            0,
                            0,  // ChunkSize (must be updated later)
                            'W'.toByte(),
                            'A'.toByte(),
                            'V'.toByte(),
                            'E'.toByte(),  // Format
                            // fmt subchunk
                            'f'.toByte(),
                            'm'.toByte(),
                            't'.toByte(),
                            ' '.toByte(),  // Subchunk1ID
                            16,
                            0,
                            0,
                            0,  // Subchunk1Size
                            1,
                            0,  // AudioFormat
                            littleBytes[0],
                            littleBytes[1],  // NumChannels
                            littleBytes[2],
                            littleBytes[3],
                            littleBytes[4],
                            littleBytes[5],  // SampleRate
                            littleBytes[6],
                            littleBytes[7],
                            littleBytes[8],
                            littleBytes[9],  // ByteRate
                            littleBytes[10],
                            littleBytes[11],  // BlockAlign
                            littleBytes[12],
                            littleBytes[13],  // BitsPerSample
                            // data subchunk
                            'd'.toByte(),
                            'a'.toByte(),
                            't'.toByte(),
                            'a'.toByte(),  // Subchunk2ID
                            0,
                            0,
                            0,
                            0
                    )
            )
        }

        @Throws(IOException::class)
        public fun updateWavHeader(wav: File) {
            val sizes = ByteBuffer
                    .allocate(8)
                    .order(ByteOrder.LITTLE_ENDIAN) // There are probably a bunch of different/better ways to calculate
                    // these two given your circumstances. Cast should be safe since if the WAV is
                    // > 4 GB we've already made a terrible mistake.
                    .putInt((wav.length() - 8).toInt()) // ChunkSize
                    .putInt((wav.length() - 44).toInt()) // Subchunk2Size
                    .array()
            var accessWave: RandomAccessFile? = null
            try {
                accessWave = RandomAccessFile(wav, "rw")
                // ChunkSize
                accessWave.seek(4)
                accessWave.write(sizes, 0, 4)

                // Subchunk2Size
                accessWave.seek(40)
                accessWave.write(sizes, 4, 4)
            } catch (ex: IOException) {
                // Rethrow but we still close accessWave in our finally
                throw ex
            } finally {
                if (accessWave != null) {
                    try {
                        accessWave.close()
                    } catch (ex: IOException) {
                        //
                    }
                }
            }
        }

        override fun onCancelled(results: Array<Any>) {
            // Handling cancellations and successful runs in the same way
            onPostExecute(results)
        }

        override fun onPostExecute(results: Array<Any>) {
            var throwable: Throwable? = null
            if (results[0] is Throwable) {
                // Error
                throwable = results[0] as Throwable
                Log.e(
                        RecordWaveTask::class.java.simpleName,
                        throwable.message,
                        throwable
                )
            }

            // If we're attached to an activity
            if (ctx != null) {
                if (throwable == null) {
                    // Display final recording stats
                    //double size = (long) results[0] / 1000000.00;
                    //long time = (long) results[1] / 1000;
                    //Toast.makeText(ctx, String.format(Locale.getDefault(), "%.2f MB / %d seconds",size, time), Toast.LENGTH_LONG).show();
                    val path = "/storage/emulated/0/Download/temp.wav"
                    val f = File(path)
                    if (f.exists()) {
                        try {
//                            val mediaPlayer = MediaPlayer()
//                            mediaPlayer.setDataSource(path)
//                            mediaPlayer.prepare()
//                            mediaPlayer.start()
                            ctx?.playAudioFile(path)
                        } catch (ex: IOException) {
                            println(ex)
                        }


                        //playAudioFile(path)
                        try {
                            ctx?.doInference(path)
                        } catch (e: IOException) {
                            e.printStackTrace()
                        }
                    }
                } else {
                    // Error
                    Toast.makeText(ctx, throwable.localizedMessage, Toast.LENGTH_LONG).show()
                }
            }
        }


        companion object {
            // Configure me!
            const val AUDIO_SOURCE = MediaRecorder.AudioSource.VOICE_RECOGNITION
            const val SAMPLE_RATE = 44100 // Hz
            const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
            const val CHANNEL_MASK = AudioFormat.CHANNEL_IN_MONO
        }

        init {
            setContext(ctx)
        }


    }

    fun playAudioFile(filename: String?) {

        try {
            val mediaPlayer = MediaPlayer()
            mediaPlayer.setDataSource(filename)
            mediaPlayer.prepare()
            mediaPlayer.start()
        } catch (ex: IOException) {
            println(ex)
        }
    }

    @Throws(Exception::class)
    fun getSplittedChunks(audioFilePath: File?): Array<File?>? {
        val SPLIT_FILE_LENGTH_MS = 10000
        val outputChunks = arrayOfNulls<File>(1000)

        //WavFileSplitter readWavFile = openWavFile(new File(filename));
        val inputWavFile = WavFileSplitter.openWavFile(audioFilePath)
        // String.format(mDuration1.toString())
        // save filename + chunk1.wav names as final chunk names
        // example: PinkPanther30-chunk1.wav, filename pinkpanther30, chunk1
        var pathStr = audioFilePath.toString()
        var delimiter1 = "/"
        var delimiter2 = ".wav"
        //val string: String = "leo_Ana_John"
        //val yourArray: List<String> = string.split("_")

        var delimiter1Splitter: List<String> = pathStr.split(delimiter1)
        Log.i("file name", delimiter1Splitter.toString())
        this.to(delimiter1Splitter)

        var delimiter1Out = delimiter1Splitter.last()
        Log.i("file name", delimiter1Out.toString())
        this.to(delimiter1Out)

        var delimiter2Splitter: List<String> = delimiter1Out.split(delimiter2)
        Log.i("file name", delimiter2Splitter.toString())
        this.to(delimiter2Splitter)

        var delimiter2Out = delimiter2Splitter.first()
        Log.i("file name", delimiter2Out.toString())
        this.to(delimiter2Out)

        // Get the number of audio channels in the wav file
        val numChannels = inputWavFile.numChannels
        // set the maximum number of frames for a target file,
        // based on the number of milliseconds assigned for each file
        val maxFramesPerFile = inputWavFile.sampleRate.toInt() * SPLIT_FILE_LENGTH_MS / 1000

        // Create a buffer of maxFramesPerFile frames
        val buffer = DoubleArray(maxFramesPerFile * numChannels)
        var framesRead: Int
        var fileCount = 0
        do {
            // Read frames into buffer
            framesRead = inputWavFile.readFrames(buffer, maxFramesPerFile)
            Log.i("framesread", framesRead.toString())
            this.to(framesRead)

            val baseDir = Environment.getExternalStorageDirectory().absolutePath
            // path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
            //File file = new File(path, "/" + fname);
            //
            val outputWavFile = WavFileSplitter.newWavFile(
                    File(baseDir, "/audioData/$delimiter2Out-chunk$fileCount.wav"),
                    inputWavFile.numChannels,
                    framesRead.toLong(),
                    inputWavFile.validBits,
                    inputWavFile.sampleRate)
            // calculating chunks duration
            // wavFile1.numFrames.toInt()
            val numFramesOut = outputWavFile.numFrames.toInt()
            Log.i("numframes", numFramesOut.toString())
            this.to(numFramesOut)

            val numSampleRateOut = outputWavFile.sampleRate.toInt()
            Log.i("samplerate", numSampleRateOut.toString())
            this.to(numSampleRateOut)

            var mChunkDuration1: Int
            mChunkDuration1 = numFramesOut / numSampleRateOut
            // Write the buffer
            if (mChunkDuration1 != 0){
                Log.i("File Duration", String.format(mChunkDuration1.toString()))
                this.to(mChunkDuration1)
                outputWavFile.writeFrames(buffer, framesRead)
                outputWavFile.close()
                val out = File(outputWavFile.toString())
                outputChunks[fileCount] = out
                fileCount++
            }
            else{
                outputWavFile.close()
                // delete last empty chunk written
                //WavFile.openWavFile(File(audioFilePath))
                val outputWavFile1 = File(baseDir+ "/audioData/" + delimiter2Out+ "-chunk"+ fileCount+ ".wav")
                //Log.i("remove last chunk", outputWavFile1.toString())
                //this.to(outputWavFile1)
                //val chunkPath = Paths.get(outputWavFile1.absolutePath)
                //Files.delete(chunkPath)
                outputWavFile1.delete()
            }
            // System.out.printf("%d %d\n", framesRead, outputWavFile);
            //fileCount++
        } while (framesRead != 0)
        run { inputWavFile.close() }
        return outputChunks
    }
}