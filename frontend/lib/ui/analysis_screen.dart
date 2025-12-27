import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../services/analysis_service.dart';
import '../services/firebase_service.dart';
import '../models/detection_result.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> with TickerProviderStateMixin {
  CameraController? _cameraController;
  final AnalysisService _analysisService = AnalysisService();
  final FirebaseService _firebaseService = FirebaseService();
  final ImagePicker _imagePicker = ImagePicker();
  
  bool _isInitialized = false;
  bool _isLoading = false;
  bool _showResult = false;
  AnalysisResult? _analysisResult;
  File? _capturedImage;
  String? _errorMessage;
  
  late AnimationController _typewriterController;
  String _displayedCaption = '';

  @override
  void initState() {
    super.initState();
    _typewriterController = AnimationController(vsync: this);
    _initializeCamera();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _typewriterController.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _errorMessage = 'Camera permission denied');
      return;
    }

    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _errorMessage = 'No cameras available');
        return;
      }

      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        backCamera,
        ResolutionPreset.max,  // Changed from high to max for better detection
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _cameraController!.initialize();
      await _analysisService.initialize();
      setState(() => _isInitialized = true);
    } catch (e) {
      setState(() => _errorMessage = 'Failed to initialize camera: $e');
    }
  }

  Future<void> _captureAndAnalyze() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    setState(() => _isLoading = true);

    try {
      final XFile file = await _cameraController!.takePicture();
      final imageFile = File(file.path);
      
      setState(() => _capturedImage = imageFile);
      
      final result = await _analysisService.analyzeImage(imageFile);
      
      if (result != null) {
        setState(() {
          _analysisResult = result;
          _showResult = true;
          _isLoading = false;
        });
        _animateTypewriter(result.caption);
        
        // Save to Firebase history (background)
        _firebaseService.saveCaption(
          imageFile: imageFile,
          caption: result.caption,
          confidence: result.captionConfidence,
          mode: 'analysis',
        ).catchError((e) => print('Failed to save to history: $e'));
      } else {
        setState(() {
          _errorMessage = 'Analysis failed';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _pickAndAnalyze() async {
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 800,   // Limit size for Firebase Base64 storage
        maxHeight: 800,
        imageQuality: 70,  // Compress to reduce file size
      );
      
      if (pickedFile == null) return;

      setState(() {
        _isLoading = true;
        _capturedImage = File(pickedFile.path);
      });

      final result = await _analysisService.analyzeImage(_capturedImage!);
      
      if (result != null) {
        setState(() {
          _analysisResult = result;
          _showResult = true;
          _isLoading = false;
        });
        _animateTypewriter(result.caption);
        
        // Save to Firebase history (background)
        _firebaseService.saveCaption(
          imageFile: _capturedImage!,
          caption: result.caption,
          confidence: result.captionConfidence,
          mode: 'analysis',
        ).catchError((e) => print('Failed to save to history: $e'));
      } else {
        setState(() {
          _errorMessage = 'Analysis failed';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  void _animateTypewriter(String caption) {
    _typewriterController.reset();
    final totalChars = caption.length;
    _typewriterController.duration = Duration(milliseconds: totalChars * 40);
    
    _typewriterController.addListener(() {
      final charCount = (_typewriterController.value * totalChars).floor();
      if (mounted) {
        setState(() => _displayedCaption = caption.substring(0, charCount));
      }
    });
    
    _typewriterController.forward();
  }

  void _resetCapture() {
    setState(() {
      _showResult = false;
      _analysisResult = null;
      _capturedImage = null;
      _displayedCaption = '';
    });
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
        title: const Text(
          'AI Analysis',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera/Image Preview
          if (_showResult && _capturedImage != null)
            Image.file(_capturedImage!, fit: BoxFit.cover)
          else if (_isInitialized && _cameraController != null)
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
            Center(child: Text(_errorMessage!, style: const TextStyle(color: Colors.white)))
          else
            const Center(child: CircularProgressIndicator(color: Colors.white)),

          // Detection boxes overlay
          if (_showResult && _analysisResult != null && _capturedImage != null)
            _buildDetectionOverlay(),

          // Gradient overlay
          Positioned(
            left: 0, right: 0, bottom: 0,
            child: Container(
              height: 400,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.transparent, Colors.black.withOpacity(0.95)],
                ),
              ),
            ),
          ),

          // Loading indicator
          if (_isLoading)
            const Center(child: CircularProgressIndicator(color: Colors.white)),

          // Results panel
          if (_showResult && _analysisResult != null)
            Positioned(
              left: 16, right: 16, bottom: 120,
              child: _buildResultsPanel(),
            ),

          // Bottom controls
          Positioned(
            left: 0, right: 0, bottom: 0,
            child: _buildBottomControls(),
          ),
        ],
      ),
    );
  }

  Widget _buildDetectionOverlay() {
    return CustomPaint(
      size: Size.infinite,
      painter: DetectionBoxPainter(
        detections: _analysisResult!.detections,
        imageSize: Size(
          _capturedImage!.readAsBytesSync().length.toDouble(),
          _capturedImage!.readAsBytesSync().length.toDouble(),
        ),
      ),
    );
  }

  Widget _buildResultsPanel() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Caption
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text('CAPTION', style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(width: 8),
              Text('${(_analysisResult!.captionConfidence * 100).toInt()}%', 
                style: TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 12)),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            _displayedCaption.isEmpty ? _analysisResult!.caption : _displayedCaption,
            style: const TextStyle(color: Colors.white, fontSize: 18, height: 1.4),
          ),
          
          const SizedBox(height: 16),
          
          // Detected objects
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.teal,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text('${_analysisResult!.detectionCount} OBJECTS', 
                  style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 4,
            children: _analysisResult!.detections.take(5).map((d) => Chip(
              label: Text('${d.className} ${(d.score * 100).toInt()}%'),
              backgroundColor: Colors.white.withOpacity(0.1),
              labelStyle: const TextStyle(color: Colors.white, fontSize: 12),
            )).toList(),
          ),
          
          const SizedBox(height: 12),
          Text(
            '${_analysisResult!.inferenceTimeMs.toInt()} ms',
            style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 12),
          ),
        ],
      ),
    ).animate().fadeIn().slideY(begin: 0.1, end: 0);
  }

  Widget _buildBottomControls() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // Gallery button
          IconButton(
            onPressed: _isLoading ? null : _pickAndAnalyze,
            icon: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(Icons.photo_library, color: Colors.white, size: 28),
            ),
          ),
          
          // Capture button
          GestureDetector(
            onTap: _isLoading ? null : (_showResult ? _resetCapture : _captureAndAnalyze),
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 4),
                color: _showResult ? Colors.white.withOpacity(0.3) : Colors.transparent,
              ),
              child: Center(
                child: _showResult
                    ? const Icon(Icons.refresh, color: Colors.white, size: 36)
                    : Container(
                        width: 64,
                        height: 64,
                        decoration: const BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: LinearGradient(
                            colors: [Color(0xFF8B5CF6), Color(0xFF3B82F6)],
                          ),
                        ),
                      ),
              ),
            ),
          ),
          
          // Placeholder for symmetry
          const SizedBox(width: 52),
        ],
      ),
    );
  }
}

class DetectionBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;

  DetectionBoxPainter({required this.detections, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final colors = [
      Colors.blue, Colors.green, Colors.orange, Colors.purple, Colors.pink,
    ];

    for (int i = 0; i < detections.length; i++) {
      final d = detections[i];
      paint.color = colors[i % colors.length];
      
      // Scale box to screen size (simplified - needs proper scaling)
      final rect = Rect.fromLTRB(
        d.left * size.width / 800,
        d.top * size.height / 600,
        d.right * size.width / 800,
        d.bottom * size.height / 600,
      );
      
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
