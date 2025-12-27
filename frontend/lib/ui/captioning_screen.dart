import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shimmer/shimmer.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:path_provider/path_provider.dart';
import '../services/api_service.dart';
import '../services/firebase_service.dart';
import '../models/caption_result.dart';

/// Image captioning screen with camera and typewriter animation
class CaptioningScreen extends StatefulWidget {
  const CaptioningScreen({super.key});

  @override
  State<CaptioningScreen> createState() => _CaptioningScreenState();
}

class _CaptioningScreenState extends State<CaptioningScreen> with TickerProviderStateMixin {
  CameraController? _cameraController;
  final ApiService _apiService = ApiService();
  final FirebaseService _firebaseService = FirebaseService();
  final ImagePicker _imagePicker = ImagePicker();
  
  bool _isInitialized = false;
  bool _isLoading = false;
  bool _showResult = false;
  CaptionResult? _captionResult;
  File? _capturedImage;
  String? _errorMessage;
  String _displayedCaption = '';
  
  // Animation controllers
  late AnimationController _typewriterController;

  @override
  void initState() {
    super.initState();
    _typewriterController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    );
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
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      
      setState(() => _isInitialized = true);
    } catch (e) {
      setState(() => _errorMessage = 'Failed to initialize camera: $e');
    }
  }

  Future<void> _captureAndCaption() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() {
      _isLoading = true;
      _showResult = false;
      _captionResult = null;
      _displayedCaption = '';
    });

    try {
      // Capture image
      final xFile = await _cameraController!.takePicture();
      final imageFile = File(xFile.path);
      
      setState(() => _capturedImage = imageFile);

      // Send to API
      final result = await _apiService.captionImage(imageFile);
      
      setState(() {
        _captionResult = result;
        _isLoading = false;
        _showResult = true;
      });

      // Start typewriter animation
      _animateTypewriter(result.caption);
      
      // Save to Firebase history (background, don't wait)
      _firebaseService.saveCaption(
        imageFile: imageFile,
        caption: result.caption,
        confidence: result.confidence,
        mode: 'captioning',
      ).catchError((e) => print('Failed to save to history: $e'));
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = e is ApiException ? e.message : 'Failed to generate caption';
      });
      
      // Show error snackbar
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(_errorMessage!),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _pickFromGallery() async {
    final xFile = await _imagePicker.pickImage(source: ImageSource.gallery);
    if (xFile == null) return;

    setState(() {
      _isLoading = true;
      _showResult = false;
      _captionResult = null;
      _displayedCaption = '';
      _capturedImage = File(xFile.path);
    });

    try {
      final result = await _apiService.captionImage(_capturedImage!);
      
      setState(() {
        _captionResult = result;
        _isLoading = false;
        _showResult = true;
      });

      _animateTypewriter(result.caption);
      
      // Save to Firebase history (background)
      _firebaseService.saveCaption(
        imageFile: _capturedImage!,
        caption: result.caption,
        confidence: result.confidence,
        mode: 'captioning',
      ).catchError((e) => print('Failed to save to history: $e'));
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = e is ApiException ? e.message : 'Failed to generate caption';
      });
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_errorMessage!), backgroundColor: Colors.red),
      );
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
      _capturedImage = null;
      _captionResult = null;
      _displayedCaption = '';
    });
    _typewriterController.reset();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _apiService.dispose();
    _typewriterController.dispose();
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
        title: const Text(
          'Image Captioning',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview or Captured Image
          if (_showResult && _capturedImage != null)
            Image.file(
              _capturedImage!,
              fit: BoxFit.cover,
            ).animate().fadeIn()
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
            _buildErrorState()
          else
            const Center(child: CircularProgressIndicator(color: Colors.white)),

          // Gradient overlay at bottom
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              height: 350,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withOpacity(0.9),
                    Colors.black,
                  ],
                ),
              ),
            ),
          ),

          // Loading Shimmer
          if (_isLoading)
            Positioned(
              left: 24,
              right: 24,
              bottom: 150,
              child: _buildLoadingShimmer(),
            ),

          // Caption Result
          if (_showResult && _captionResult != null)
            Positioned(
              left: 24,
              right: 24,
              bottom: 150,
              child: _buildCaptionResult(),
            ),

          // Bottom Controls
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: _buildBottomControls(),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 64),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              style: const TextStyle(color: Colors.white),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: _pickFromGallery,
              child: const Text('Pick from Gallery'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingShimmer() {
    return Shimmer.fromColors(
      baseColor: Colors.grey[800]!,
      highlightColor: Colors.grey[600]!,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(height: 20, width: double.infinity, color: Colors.white),
            const SizedBox(height: 12),
            Container(height: 20, width: 200, color: Colors.white),
            const SizedBox(height: 12),
            Container(height: 14, width: 100, color: Colors.white),
          ],
        ),
      ),
    );
  }

  Widget _buildCaptionResult() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 20,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Caption with typewriter effect
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.auto_awesome,
                color: Colors.amber,
                size: 24,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  _displayedCaption,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    height: 1.4,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          
          // Confidence and timing
          Row(
            children: [
              _buildStatChip(
                icon: Icons.speed,
                label: '${_captionResult!.inferenceTimeMs.toStringAsFixed(0)}ms',
              ),
              const SizedBox(width: 8),
              _buildStatChip(
                icon: Icons.check_circle,
                label: '${(_captionResult!.confidence * 100).toStringAsFixed(0)}% confidence',
              ),
            ],
          ),
        ],
      ),
    ).animate().fadeIn(duration: 300.ms).slideY(begin: 0.1, end: 0);
  }

  Widget _buildStatChip({required IconData icon, required String label}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: Colors.white70, size: 14),
          const SizedBox(width: 4),
          Text(
            label,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomControls() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Gallery button
            _buildControlButton(
              icon: Icons.photo_library,
              onPressed: _pickFromGallery,
              isSmall: true,
            ),
            
            // Capture button
            if (_showResult)
              _buildControlButton(
                icon: Icons.refresh,
                onPressed: _resetCapture,
                isPrimary: true,
              )
            else
              _buildControlButton(
                icon: Icons.camera,
                onPressed: _isLoading ? null : _captureAndCaption,
                isPrimary: true,
              ),
            
            // Placeholder for symmetry
            const SizedBox(width: 56),
          ],
        ),
      ),
    );
  }

  Widget _buildControlButton({
    required IconData icon,
    VoidCallback? onPressed,
    bool isPrimary = false,
    bool isSmall = false,
  }) {
    final size = isPrimary ? 72.0 : (isSmall ? 48.0 : 56.0);
    
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: isPrimary
              ? const LinearGradient(
                  colors: [Color(0xFFF59E0B), Color(0xFFEF4444)],
                )
              : null,
          color: isPrimary ? null : Colors.white.withOpacity(0.2),
          border: isPrimary
              ? Border.all(color: Colors.white, width: 3)
              : null,
        ),
        child: Icon(
          icon,
          color: Colors.white,
          size: isPrimary ? 32 : 24,
        ),
      ),
    );
  }
}
