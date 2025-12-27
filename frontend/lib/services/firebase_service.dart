import 'dart:convert';
import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
// Firebase Storage removed - using Base64 in Firestore instead

/// History entry for storing captions/detections in Firebase
class HistoryEntry {
  final String id;
  final String imageUrl;
  final String caption;
  final double confidence;
  final DateTime timestamp;
  final String mode; // 'captioning' or 'detection'
  final String? userId;

  HistoryEntry({
    required this.id,
    required this.imageUrl,
    required this.caption,
    required this.confidence,
    required this.timestamp,
    required this.mode,
    this.userId,
  });

  factory HistoryEntry.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return HistoryEntry(
      id: doc.id,
      imageUrl: data['imageUrl'] ?? '',
      caption: data['caption'] ?? '',
      confidence: (data['confidence'] as num?)?.toDouble() ?? 0.0,
      timestamp: (data['timestamp'] as Timestamp?)?.toDate() ?? DateTime.now(),
      mode: data['mode'] ?? 'captioning',
      userId: data['userId'],
    );
  }

  Map<String, dynamic> toFirestore() => {
        'imageUrl': imageUrl,
        'caption': caption,
        'confidence': confidence,
        'timestamp': Timestamp.fromDate(timestamp),
        'mode': mode,
        'userId': userId,
      };
}

/// Service for Firebase Firestore operations (no Storage required)
class FirebaseService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  // Storage removed - images stored as Base64 in Firestore
  
  /// Collection name for caption history
  static const String collectionName = 'captions';
  
  /// Device ID for anonymous user tracking
  String? _deviceId;
  
  /// Set the device ID for anonymous user tracking
  void setDeviceId(String deviceId) {
    _deviceId = deviceId;
  }
  
  /// Convert image to Base64 string
  /// 
  /// Returns a data URL that can be displayed directly
  Future<String> imageToBase64(File imageFile) async {
    // Read image bytes
    final bytes = await imageFile.readAsBytes();
    
    // Convert to Base64
    final base64String = base64Encode(bytes);
    
    // If image is too large for Firestore (> 900KB), return placeholder
    // Firestore document limit is ~1MB, but we leave room for other fields
    if (base64String.length > 900000) {
      print('Image too large for Firestore (${base64String.length} chars), using placeholder');
      return 'data:image/png;base64,';  // Empty placeholder
    }
    
    print('Image Base64 size: ${base64String.length} chars');
    // Return as data URL
    return 'data:image/jpeg;base64,$base64String';
  }
  
  /// Save a history entry to Firestore
  /// 
  /// [entry] - The history entry to save
  /// 
  /// Returns the document ID of the created entry
  Future<String> saveHistoryEntry(HistoryEntry entry) async {
    final doc = await _firestore.collection(collectionName).add(
      entry.toFirestore()..['userId'] = _deviceId,
    );
    return doc.id;
  }
  
  /// Save a caption with Base64 image encoding
  /// 
  /// [imageFile] - The image file to convert and store
  /// [caption] - The generated caption
  /// [confidence] - Confidence score
  /// 
  /// Returns the document ID
  Future<String> saveCaption({
    required File imageFile,
    required String caption,
    required double confidence,
    String mode = 'captioning',
  }) async {
    print('📝 [FirebaseService] saveCaption called');
    print('   Caption: $caption');
    print('   Confidence: $confidence');
    print('   Mode: $mode');
    
    try {
      // Convert image to Base64 (no Storage needed!)
      print('   Converting image to Base64...');
      final imageUrl = await imageToBase64(imageFile);
      print('   Base64 length: ${imageUrl.length} chars');
      
      // Create and save entry
      final entry = HistoryEntry(
        id: '',
        imageUrl: imageUrl,
        caption: caption,
        confidence: confidence,
        timestamp: DateTime.now(),
        mode: mode,
        userId: _deviceId,
      );
      
      print('   Saving to Firestore collection: $collectionName');
      final docId = await saveHistoryEntry(entry);
      print('✅ [FirebaseService] Saved successfully! Doc ID: $docId');
      return docId;
    } catch (e, stackTrace) {
      print('❌ [FirebaseService] SAVE FAILED: $e');
      print('   Stack trace: $stackTrace');
      rethrow;  // Let caller handle the error
    }
  }
  
  /// Get paginated history entries
  /// 
  /// [limit] - Number of entries per page
  /// [startAfter] - Document to start after for pagination
  /// 
  /// Returns a list of history entries
  Future<List<HistoryEntry>> getHistory({
    int limit = 20,
    DocumentSnapshot? startAfter,
  }) async {
    Query query = _firestore
        .collection(collectionName)
        .orderBy('timestamp', descending: true)
        .limit(limit);
    
    // Filter by user if deviceId is set
    if (_deviceId != null) {
      query = query.where('userId', isEqualTo: _deviceId);
    }
    
    // Pagination
    if (startAfter != null) {
      query = query.startAfterDocument(startAfter);
    }
    
    final snapshot = await query.get();
    return snapshot.docs.map((doc) => HistoryEntry.fromFirestore(doc)).toList();
  }
  
  /// Delete a history entry (image is stored as Base64 in document)
  Future<void> deleteEntry(String entryId, String imageUrl) async {
    // Just delete from Firestore - image is stored as Base64 in the document
    await _firestore.collection(collectionName).doc(entryId).delete();
  }
  
  /// Stream of history entries for real-time updates
  Stream<List<HistoryEntry>> historyStream({int limit = 20}) {
    Query query = _firestore
        .collection(collectionName)
        .orderBy('timestamp', descending: true)
        .limit(limit);
    
    if (_deviceId != null) {
      query = query.where('userId', isEqualTo: _deviceId);
    }
    
    return query.snapshots().map((snapshot) =>
        snapshot.docs.map((doc) => HistoryEntry.fromFirestore(doc)).toList());
  }
}
