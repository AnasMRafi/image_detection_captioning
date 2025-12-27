import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:shimmer/shimmer.dart';
import '../services/firebase_service.dart';
import 'package:intl/intl.dart';

/// History screen showing saved captions with pagination
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final FirebaseService _firebaseService = FirebaseService();
  final List<HistoryEntry> _entries = [];
  bool _isLoading = true;
  bool _hasMore = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _isLoading = true);
    
    try {
      final entries = await _firebaseService.getHistory(limit: 20);
      setState(() {
        _entries.clear();
        _entries.addAll(entries);
        _hasMore = entries.length == 20;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Firebase not configured. History requires Firebase setup.';
      });
    }
  }

  Future<void> _loadMore() async {
    if (!_hasMore || _isLoading) return;
    
    // Pagination would use the last document - simplified for demo
    setState(() => _isLoading = true);
    
    try {
      final entries = await _firebaseService.getHistory(limit: 20);
      setState(() {
        _entries.addAll(entries);
        _hasMore = entries.length == 20;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _deleteEntry(HistoryEntry entry, int index) async {
    // Remove from list immediately for responsive UI
    setState(() => _entries.removeAt(index));

    try {
      await _firebaseService.deleteEntry(entry.id, entry.imageUrl);
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('Entry deleted'),
            action: SnackBarAction(
              label: 'Undo',
              onPressed: () {
                setState(() => _entries.insert(index, entry));
              },
            ),
          ),
        );
      }
    } catch (e) {
      // Restore on error
      setState(() => _entries.insert(index, entry));
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to delete'), backgroundColor: Colors.red),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'History',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadHistory,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _errorMessage != null
          ? _buildErrorState()
          : _entries.isEmpty && !_isLoading
              ? _buildEmptyState()
              : _buildHistoryList(),
    );
  }

  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.cloud_off,
              size: 80,
              color: Colors.grey[600],
            ),
            const SizedBox(height: 24),
            Text(
              _errorMessage!,
              style: TextStyle(
                color: Colors.grey[400],
                fontSize: 16,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Text(
              'Set up Firebase in your Flutter project to enable history.',
              style: TextStyle(
                color: Colors.grey[600],
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.history,
            size: 80,
            color: Colors.grey[600],
          ),
          const SizedBox(height: 24),
          Text(
            'No history yet',
            style: TextStyle(
              color: Colors.grey[400],
              fontSize: 18,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Your captions will appear here',
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 14,
            ),
          ),
        ],
      ),
    ).animate().fadeIn();
  }

  Widget _buildHistoryList() {
    return RefreshIndicator(
      onRefresh: _loadHistory,
      color: Colors.deepPurple,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _entries.length + (_hasMore ? 1 : 0),
        itemBuilder: (context, index) {
          if (index == _entries.length) {
            // Load more indicator
            if (_isLoading) {
              return const Center(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: CircularProgressIndicator(color: Colors.white),
                ),
              );
            }
            return const SizedBox.shrink();
          }

          final entry = _entries[index];
          return _HistoryCard(
            entry: entry,
            onDelete: () => _deleteEntry(entry, index),
            onTap: () => _showFullImage(entry),
          ).animate(delay: Duration(milliseconds: 50 * index))
            .fadeIn(duration: 300.ms)
            .slideX(begin: 0.1, end: 0);
        },
      ),
    );
  }

  void _showFullImage(HistoryEntry entry) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.9,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, scrollController) => Container(
          decoration: const BoxDecoration(
            color: Color(0xFF1A1D21),
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              // Handle
              Container(
                margin: const EdgeInsets.only(top: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[600],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              
              // Image
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: _buildFullImage(entry.imageUrl),
                  ),
                ),
              ),
              
              // Caption
              Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.caption,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Icon(Icons.check_circle, size: 16, color: Colors.green[400]),
                        const SizedBox(width: 6),
                        Text(
                          '${(entry.confidence * 100).toStringAsFixed(0)}% confidence',
                          style: TextStyle(color: Colors.grey[400]),
                        ),
                        const Spacer(),
                        Text(
                          DateFormat('MMM d, y • h:mm a').format(entry.timestamp),
                          style: TextStyle(color: Colors.grey[500], fontSize: 12),
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
    );
  }

  /// Build full image that handles both Base64 data URLs and network URLs
  Widget _buildFullImage(String imageUrl) {
    // Empty or invalid URL
    if (imageUrl.isEmpty || imageUrl.length < 30) {
      return const Center(
        child: Icon(Icons.image, size: 80, color: Colors.grey),
      );
    }
    
    // Base64 data URL
    if (imageUrl.startsWith('data:image')) {
      try {
        final base64Part = imageUrl.split(',').last;
        if (base64Part.isEmpty) {
          return const Center(
            child: Icon(Icons.image, size: 80, color: Colors.grey),
          );
        }
        final bytes = base64Decode(base64Part);
        return Image.memory(
          bytes,
          fit: BoxFit.contain,
          errorBuilder: (_, __, ___) => const Center(
            child: Icon(Icons.broken_image, size: 80, color: Colors.grey),
          ),
        );
      } catch (e) {
        return const Center(
          child: Icon(Icons.broken_image, size: 80, color: Colors.grey),
        );
      }
    }
    
    // Network URL (fallback for old entries)
    return Image.network(
      imageUrl,
      fit: BoxFit.contain,
      errorBuilder: (_, __, ___) => const Center(
        child: Icon(Icons.broken_image, size: 80, color: Colors.grey),
      ),
    );
  }
}

/// Card widget for a single history entry
class _HistoryCard extends StatelessWidget {
  final HistoryEntry entry;
  final VoidCallback onDelete;
  final VoidCallback onTap;

  const _HistoryCard({
    required this.entry,
    required this.onDelete,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Dismissible(
      key: Key(entry.id),
      direction: DismissDirection.endToStart,
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 24),
        decoration: BoxDecoration(
          color: Colors.red,
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Icon(Icons.delete, color: Colors.white),
      ),
      onDismissed: (_) => onDelete(),
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white10),
          ),
          child: Row(
            children: [
              // Thumbnail with shimmer loading
              ClipRRect(
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  bottomLeft: Radius.circular(16),
                ),
                child: SizedBox(
                  width: 100,
                  height: 100,
                  child: _buildThumbnail(entry.imageUrl),
                ),
              ),
              
              // Content
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        entry.caption,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 2,
                            ),
                            decoration: BoxDecoration(
                              color: entry.mode == 'captioning'
                                  ? Colors.orange.withOpacity(0.2)
                                  : Colors.blue.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              entry.mode,
                              style: TextStyle(
                                color: entry.mode == 'captioning'
                                    ? Colors.orange
                                    : Colors.blue,
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                          const Spacer(),
                          Text(
                            _formatTimeAgo(entry.timestamp),
                            style: TextStyle(
                              color: Colors.grey[500],
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              
              // Arrow
              Padding(
                padding: const EdgeInsets.only(right: 12),
                child: Icon(
                  Icons.chevron_right,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Build thumbnail image that handles both Base64 data URLs and network URLs
  Widget _buildThumbnail(String imageUrl) {
    // Empty or invalid URL
    if (imageUrl.isEmpty || imageUrl.length < 30) {
      return Container(
        color: Colors.grey[800],
        child: const Icon(Icons.image, color: Colors.grey, size: 40),
      );
    }
    
    // Base64 data URL
    if (imageUrl.startsWith('data:image')) {
      try {
        // Extract base64 part after the comma
        final base64Part = imageUrl.split(',').last;
        if (base64Part.isEmpty) {
          return Container(
            color: Colors.grey[800],
            child: const Icon(Icons.image, color: Colors.grey, size: 40),
          );
        }
        final bytes = base64Decode(base64Part);
        return Image.memory(
          bytes,
          fit: BoxFit.cover,
          errorBuilder: (_, __, ___) => Container(
            color: Colors.grey[800],
            child: const Icon(Icons.broken_image, color: Colors.grey),
          ),
        );
      } catch (e) {
        return Container(
          color: Colors.grey[800],
          child: const Icon(Icons.broken_image, color: Colors.grey),
        );
      }
    }
    
    // Network URL (fallback for old entries)
    return Image.network(
      imageUrl,
      fit: BoxFit.cover,
      errorBuilder: (_, __, ___) => Container(
        color: Colors.grey[800],
        child: const Icon(Icons.broken_image, color: Colors.grey),
      ),
    );
  }

  String _formatTimeAgo(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);

    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return DateFormat('MMM d').format(date);
  }
}
