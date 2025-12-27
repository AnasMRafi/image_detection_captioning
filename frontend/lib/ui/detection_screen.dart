import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/detection_service.dart';
import '../models/detection_result.dart';

/// Real-time object detection screen with camera preview
/// Uses server-side Faster R-CNN for detection
class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  CameraController? _cameraController;
  DetectionService? _detectionService;
  List<DetectionResult> _detections = [];
  bool _isInitialized = false;
  bool _isDetecting = true;
  bool _isProcessing = false;
  double _confidenceThreshold = 0.5;
  double _inferenceTime = 0;
  String? _errorMessage;
  Timer? _detectionTimer;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    // Request camera permission
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _errorMessage = 'Camera permission denied');
      return;
    }

    try {
      // Get available cameras
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _errorMessage = 'No cameras available');
        return;
      }

      // Use back camera
      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      // Initialize camera controller
      _cameraController = CameraController(
        backCamera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();

      // Initialize detection service
      _detectionService = DetectionService();
      final serviceReady = await _detectionService!.initialize();
      
      if (!serviceReady) {
        setState(() => _errorMessage = 
          'Detection server not available. Check Settings (⚙️) for correct IP.');
      }

      setState(() => _isInitialized = true);
      
      // Start periodic detection (capture frames every 500ms)
      if (serviceReady) {
        _startPeriodicDetection();
      }
    } catch (e) {
      setState(() => _errorMessage = 'Failed to initialize camera: $e');
    }
  }

  void _startPeriodicDetection() {
    _detectionTimer?.cancel();
    // Reduced from 500ms to 200ms for faster detection
    _detectionTimer = Timer.periodic(const Duration(milliseconds: 200), (_) {
      if (_isDetecting && !_isProcessing && mounted) {
        _captureAndDetect();
      }
    });
  }

  Future<void> _captureAndDetect() async {
    if (_cameraController == null || 
        !_cameraController!.value.isInitialized ||
        _detectionService == null ||
        !_detectionService!.isInitialized) {
      return;
    }

    _isProcessing = true;

    try {
      // Capture frame as JPEG
      final XFile imageFile = await _cameraController!.takePicture();
      final Uint8List bytes = await imageFile.readAsBytes();
      
      // Send to server for detection
      final startTime = DateTime.now();
      final results = await _detectionService!.detectFromBytes(
        bytes,
        filename: 'frame.jpg',
        confidenceThreshold: _confidenceThreshold,
      );
      final elapsed = DateTime.now().difference(startTime).inMilliseconds;

      if (mounted) {
        setState(() {
          _detections = results;
          _inferenceTime = elapsed.toDouble();
        });
      }
    } catch (e) {
      print('Detection error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  void _toggleDetection() {
    setState(() => _isDetecting = !_isDetecting);
    if (!_isDetecting) {
      setState(() => _detections = []);
    }
  }

  void _updateConfidenceThreshold(double value) {
    setState(() => _confidenceThreshold = value);
  }

  @override
  void dispose() {
    _detectionTimer?.cancel();
    _cameraController?.dispose();
    _detectionService?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.arrow_back, color: Colors.white),
          ),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          // Inference Time Badge
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.speed,
                  color: _inferenceTime < 500 ? Colors.greenAccent : Colors.orange,
                  size: 16,
                ),
                const SizedBox(width: 6),
                Text(
                  '${_inferenceTime.toInt()} ms',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          if (_isInitialized && _cameraController != null)
            SizedBox.expand(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _cameraController!.value.previewSize!.height,
                  height: _cameraController!.value.previewSize!.width,
                  child: CameraPreview(_cameraController!),
                ),
              ),
            )
          else if (_errorMessage != null)
            Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.error_outline,
                      color: Colors.red,
                      size: 64,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      _errorMessage!,
                      style: const TextStyle(color: Colors.white),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            )
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),

          // Bounding Box Overlay
          if (_isInitialized && _cameraController != null)
            CustomPaint(
              painter: BoundingBoxPainter(
                detections: _detections,
                previewSize: _cameraController!.value.previewSize ?? const Size(1, 1),
                screenSize: MediaQuery.of(context).size,
              ),
            ),

          // Processing Indicator
          if (_isProcessing)
            Positioned(
              top: 100,
              left: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.8),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  children: [
                    SizedBox(
                      width: 12, height: 12,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    ),
                    SizedBox(width: 8),
                    Text('Detecting...', style: TextStyle(color: Colors.white)),
                  ],
                ),
              ),
            ),

          // Bottom Controls
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: _buildBottomControls(),
          ),

          // Detection Count Badge
          if (_detections.isNotEmpty)
            Positioned(
              top: 100,
              right: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${_detections.length} objects',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildBottomControls() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.transparent,
            Colors.black.withOpacity(0.8),
          ],
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Confidence Threshold Slider
          Row(
            children: [
              const Text(
                'Confidence',
                style: TextStyle(color: Colors.white70, fontSize: 12),
              ),
              Expanded(
                child: Slider(
                  value: _confidenceThreshold,
                  min: 0.1,
                  max: 0.9,
                  divisions: 8,
                  activeColor: Colors.deepPurple,
                  inactiveColor: Colors.white24,
                  label: '${(_confidenceThreshold * 100).toInt()}%',
                  onChanged: _updateConfidenceThreshold,
                ),
              ),
              Text(
                '${(_confidenceThreshold * 100).toInt()}%',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // Toggle Button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _toggleDetection,
              icon: Icon(_isDetecting ? Icons.pause : Icons.play_arrow),
              label: Text(_isDetecting ? 'Pause Detection' : 'Resume Detection'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isDetecting ? Colors.red : Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),


        ],
      ),
    );
  }
}

/// Custom painter for drawing bounding boxes on detected objects
class BoundingBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size previewSize;
  final Size screenSize;

  // Color palette for different classes
  static const List<Color> _classColors = [
    Colors.blue,
    Colors.green,
    Colors.orange,
    Colors.purple,
    Colors.pink,
    Colors.cyan,
    Colors.amber,
    Colors.teal,
    Colors.red,
    Colors.indigo,
  ];

  BoundingBoxPainter({
    required this.detections,
    required this.previewSize,
    required this.screenSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale factors
    // Camera preview is rotated 90 degrees
    final scaleX = size.width / previewSize.height;
    final scaleY = size.height / previewSize.width;

    for (final detection in detections) {
      // Get color based on class label
      final color = _classColors[detection.label % _classColors.length];
      
      // Scale bounding box to screen coordinates
      // Note: Box coordinates are in original image space
      final scaledRect = Rect.fromLTRB(
        detection.left * scaleX,
        detection.top * scaleY,
        detection.right * scaleX,
        detection.bottom * scaleY,
      );

      // Draw bounding box
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      final rrect = RRect.fromRectAndRadius(scaledRect, const Radius.circular(8));
      canvas.drawRRect(rrect, boxPaint);

      // Draw label background
      final labelText = '${detection.className} ${detection.scorePercent}';
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelBgRect = Rect.fromLTWH(
        scaledRect.left,
        scaledRect.top - 22,
        textPainter.width + 12,
        20,
      );

      final bgPaint = Paint()
        ..color = color.withOpacity(0.8)
        ..style = PaintingStyle.fill;

      final labelRRect = RRect.fromRectAndCorners(
        labelBgRect,
        topLeft: const Radius.circular(6),
        topRight: const Radius.circular(6),
        bottomRight: const Radius.circular(0),
        bottomLeft: const Radius.circular(0),
      );
      canvas.drawRRect(labelRRect, bgPaint);

      // Draw label text
      textPainter.paint(
        canvas,
        Offset(scaledRect.left + 6, scaledRect.top - 20),
      );
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return detections != oldDelegate.detections;
  }
}
