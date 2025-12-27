import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'detection_screen.dart';
import 'captioning_screen.dart';
import 'history_screen.dart';
import 'analysis_screen.dart';

import '../services/api_service.dart';

/// Home screen with mode selection cards
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentPage = 0;
  final PageController _pageController = PageController();
  
  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Color(0xFF0D1117),
                Colors.transparent,
              ],
            ),
          ),
        ),
        title: ShaderMask(
          shaderCallback: (bounds) => const LinearGradient(
            colors: [Color(0xFF3B82F6), Color(0xFF8B5CF6)],
          ).createShader(bounds),
          child: const Text(
            'AI Vision',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 28,
              color: Colors.white,
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white70),
            onPressed: () {
              HapticFeedback.lightImpact();
              _showSettingsDialog(context);
            },
            tooltip: 'Settings',
          ),
          IconButton(
            icon: const Icon(Icons.history, color: Colors.white70),
            onPressed: () {
              HapticFeedback.lightImpact();
              _navigateToHistory(context);
            },
            tooltip: 'View History',
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0D1117),
              Color(0xFF1A1D21),
              Color(0xFF0D1117),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
              // Header Text
              Text(
                'Choose Mode',
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.white.withOpacity(0.6),
                ),
              ).animate().fadeIn(duration: 300.ms),
              
              const SizedBox(height: 16),
              
              // Mode Cards
              Expanded(
                child: Column(
                  children: [
                    // Object Detection Card
                    Expanded(
                      child: _ModeCard(
                        title: 'Object Detection',
                        subtitle: 'Real-time • YOLOv12',
                        description: 'Detect and identify objects using your camera with state-of-the-art AI.',
                        icon: Icons.crop_free_rounded,
                        gradient: const LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [Color(0xFF3B82F6), Color(0xFF8B5CF6)],
                        ),
                        onTap: () => _navigateToDetection(context),
                      ),
                    ).animate()
                      .fadeIn(delay: 100.ms, duration: 400.ms)
                      .slideY(begin: 0.1, end: 0),
                    
                    const SizedBox(height: 8),
                    
                    // Combined Analysis Card (NEW)
                    Expanded(
                      child: _ModeCard(
                        title: 'AI Analysis',
                        subtitle: 'Detection + Caption',
                        description: 'Combine detection with guided captioning for enhanced descriptions.',
                        icon: Icons.auto_fix_high_rounded,
                        gradient: const LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [Color(0xFF10B981), Color(0xFF06B6D4)],
                        ),
                        onTap: () => _navigateToAnalysis(context),
                      ),
                    ).animate()
                      .fadeIn(delay: 150.ms, duration: 400.ms)
                      .slideY(begin: 0.1, end: 0),
                    
                    const SizedBox(height: 8),
                    
                    // Image Captioning Card
                    Expanded(
                      child: _ModeCard(
                        title: 'Image Captioning',
                        subtitle: 'AI-Powered • Online',
                        description: 'Generate natural language descriptions of your photos.',
                        icon: Icons.auto_awesome_rounded,
                        gradient: const LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [Color(0xFFF59E0B), Color(0xFFEF4444)],
                        ),
                        onTap: () => _navigateToCaptioning(context),
                      ),
                    ).animate()
                      .fadeIn(delay: 200.ms, duration: 400.ms)
                      .slideY(begin: 0.1, end: 0),
                  ],
                ),
              ),
              
              const SizedBox(height: 12),
              
              // Footer info
              Center(
                child: Text(
                  'Powered by YOLOv8 & CNN-LSTM Attention',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.white.withOpacity(0.3),
                  ),
                ),
              ).animate().fadeIn(delay: 400.ms),
            ],
          ),
        ),
        ),
      ),
    );
  }


  void _showSettingsDialog(BuildContext context) {
    final TextEditingController controller = TextEditingController(
      text: ApiConfig.baseUrl.replaceAll('http://', ''),
    );
    
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1E293B),
        title: const Text('Server Configuration', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter Backend Server IP (Port 8000)',
              style: TextStyle(color: Colors.white70, fontSize: 13),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                filled: true,
                fillColor: Colors.black26,
                hintText: 'e.g., 192.168.1.50:8000',
                hintStyle: TextStyle(color: Colors.white.withOpacity(0.3)),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                prefixText: 'http://',
                prefixStyle: const TextStyle(color: Colors.white54),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'Run "ipconfig getifaddr en0" on Mac to find IP.', 
              style: TextStyle(color: Colors.white38, fontSize: 11)
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              if (controller.text.isNotEmpty) {
                await ApiConfig.setBaseUrl(controller.text);
                if (mounted) {
                  Navigator.pop(ctx);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Server URL set to: ${ApiConfig.baseUrl}')),
                  );
                }
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }
  
  void _navigateToDetection(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const DetectionScreen()),
    );
  }
  
  void _navigateToAnalysis(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const AnalysisScreen()),
    );
  }
  
  void _navigateToCaptioning(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const CaptioningScreen()),
    );
  }
  
  void _navigateToHistory(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const HistoryScreen()),
    );
  }
}

/// Animated mode selection card
class _ModeCard extends StatefulWidget {
  final String title;
  final String subtitle;
  final String description;
  final IconData icon;
  final Gradient gradient;
  final VoidCallback onTap;

  const _ModeCard({
    required this.title,
    required this.subtitle,
    required this.description,
    required this.icon,
    required this.gradient,
    required this.onTap,
  });

  @override
  State<_ModeCard> createState() => _ModeCardState();
}

class _ModeCardState extends State<_ModeCard> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) {
        setState(() => _isPressed = false);
        HapticFeedback.mediumImpact();  // Haptic feedback
        widget.onTap();
      },
      onTapCancel: () => setState(() => _isPressed = false),
      child: AnimatedScale(
        duration: const Duration(milliseconds: 150),
        scale: _isPressed ? 0.97 : 1.0,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    widget.gradient.colors.first.withOpacity(0.8),
                    widget.gradient.colors.last.withOpacity(0.6),
                  ],
                ),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.white.withOpacity(0.2),
                  width: 1.5,
                ),
                boxShadow: [
                  BoxShadow(
                    color: (widget.gradient.colors.first).withOpacity(0.3),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Stack(
                children: [
              // Background pattern
              Positioned(
                right: -20,
                bottom: -20,
                child: Icon(
                  widget.icon,
                  size: 150,
                  color: Colors.white.withOpacity(0.1),
                ),
              ),
              
              // Content
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Icon (smaller)
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Icon(
                        widget.icon,
                        color: Colors.white,
                        size: 22,
                      ),
                    ),
                    
                    const Spacer(),
                    
                    // Subtitle badge
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        widget.subtitle,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                    
                    const SizedBox(height: 4),
                    
                    // Title (smaller)
                    Text(
                      widget.title,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    
                    const SizedBox(height: 4),
                    
                    // Description (more compact)
                    Text(
                      widget.description,
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 12,
                        height: 1.3,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    
                    // Arrow hint (inline, no spacer)
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Text(
                          'Open',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.9),
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 4),
                        const Icon(
                          Icons.arrow_forward_rounded,
                          color: Colors.white,
                          size: 14,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
          ),
        ),
      ),
    );
  }
}
