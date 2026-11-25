import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'api_client.dart';

class CasetaPage extends StatefulWidget {
  final TollApiClient api;
  final int casetaId;

  const CasetaPage({super.key, required this.api, required this.casetaId});

  @override
  State<CasetaPage> createState() => _CasetaPageState();
}

class _CasetaPageState extends State<CasetaPage> {
  CameraController? _cam;
  bool _camReady = false;
  bool _streaming = false;
  bool _sending = false;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);
  Map<String, dynamic>? _lastResult;
  String _status = 'Inicializando cámara...';

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final perm = await Permission.camera.request();
    if (!perm.isGranted) {
      if (mounted) setState(() => _status = 'Permiso de cámara denegado');
      return;
    }
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        if (mounted) setState(() => _status = 'No se encontró cámara');
        return;
      }
      _cam = CameraController(
        cams.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _cam!.initialize();
      if (!mounted) return;
      setState(() {
        _camReady = true;
        _status = 'Cámara lista (2-3 fps o captura manual)';
      });
    } catch (e) {
      if (mounted) setState(() => _status = 'Error cámara: $e');
    }
  }

  @override
  void dispose() {
    _stopStream();
    _cam?.dispose();
    super.dispose();
  }

  void _toggleStream() async {
    if (!_camReady || _cam == null) return;
    if (_streaming) {
      await _stopStream();
    } else {
      setState(() {
        _streaming = true;
        _status = 'Modo continuo activo (~2 fps)...';
      });
      await _cam!.startImageStream((CameraImage image) async {
        if (!_streaming || _sending) return;
        final now = DateTime.now();
        if (now.difference(_lastSent).inMilliseconds < 450) return;
        _lastSent = now;
        _sending = true;
        try {
          final bytes = _encodeCameraImage(image);
          if (bytes != null) {
            await _procesarBytes(bytes);
          }
        } finally {
          _sending = false;
        }
      });
    }
  }

  Future<void> _stopStream() async {
    if (_cam != null && _cam!.value.isStreamingImages) {
      await _cam!.stopImageStream();
    }
    if (mounted) {
      setState(() {
        _streaming = false;
        _status = 'Captura en pausa';
      });
    }
  }

  Future<void> _capturarManual() async {
    if (!_camReady || _cam == null || _sending) return;
    final wasStreaming = _streaming;
    if (wasStreaming) await _stopStream();
    setState(() {
      _sending = true;
      _status = 'Capturando y analizando en RunPod...';
    });
    try {
      final shot = await _cam!.takePicture();
      final raw = await shot.readAsBytes();
      await _procesarBytes(raw);
    } catch (e) {
      if (mounted) setState(() => _status = 'Error: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _procesarBytes(Uint8List jpeg) async {
    try {
      final res = await widget.api.sendFrame(jpeg, widget.casetaId);
      if (!mounted) return;
      setState(() {
        _lastResult = res;
        final paso = res['paso'] as Map<String, dynamic>;
        final lat = (res['inferencia']?['latencia_ms']?['total']) ?? paso['latencia_ms'] ?? '-';
        _status = 'OK · ${paso['placa_normalizada'] ?? paso['placa_leida']} · $lat ms';
      });
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Error API: ${e.toString().replaceFirst('Exception: ', '')}');
      }
    }
  }

  Uint8List? _encodeCameraImage(CameraImage image) {
    try {
      final w = image.width;
      final h = image.height;
      final out = img.Image(width: w, height: h);
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes.length > 2 ? image.planes[2] : image.planes[1];
      final uvPixelStride = uPlane.bytesPerPixel ?? 1;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final yIdx = y * yPlane.bytesPerRow + x;
          final uvIdx = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2) * uvPixelStride;
          if (yIdx >= yPlane.bytes.length || uvIdx >= uPlane.bytes.length || uvIdx >= vPlane.bytes.length) {
            continue;
          }
          final yp = yPlane.bytes[yIdx];
          final up = uPlane.bytes[uvIdx];
          final vp = vPlane.bytes[uvIdx];
          final r = (yp + 1.402 * (vp - 128)).clamp(0, 255).toInt();
          final g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).clamp(0, 255).toInt();
          final b = (yp + 1.772 * (up - 128)).clamp(0, 255).toInt();
          out.setPixelRgba(x, y, r, g, b, 255);
        }
      }
      final resized = w > 800 ? img.copyResize(out, width: 800) : out;
      return Uint8List.fromList(img.encodeJpg(resized, quality: 75));
    } catch (_) {
      return null;
    }
  }

  Color _estadoColor(String? estado) {
    switch (estado) {
      case 'AUTORIZADO':
        return Colors.greenAccent;
      case 'NO_REGISTRADA':
        return Colors.redAccent;
      case 'SIN_SALDO':
        return Colors.amberAccent;
      default:
        return Colors.white70;
    }
  }

  @override
  Widget build(BuildContext context) {
    final paso = _lastResult?['paso'] as Map<String, dynamic>?;
    return Column(
      children: [
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (_camReady && _cam != null)
                CameraPreview(_cam!)
              else
                Center(child: Text(_status, style: const TextStyle(color: Colors.white70))),
              Positioned(
                top: 10,
                left: 10,
                right: 10,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.65),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(_status, style: const TextStyle(fontSize: 12, color: Colors.white)),
                ),
              ),
            ],
          ),
        ),
        Container(
          padding: const EdgeInsets.all(14),
          color: const Color(0xFF0F172A),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (paso != null) ...[
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      (paso['placa_normalizada'] as String?)?.isNotEmpty == true
                          ? paso['placa_normalizada'] as String
                          : (paso['placa_leida'] ?? 'SIN PLACA').toString(),
                      style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, letterSpacing: 1.5),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: _estadoColor(paso['estado'] as String?).withValues(alpha: 0.18),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: _estadoColor(paso['estado'] as String?)),
                      ),
                      child: Text(
                        (paso['estado'] ?? '-').toString(),
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: _estadoColor(paso['estado'] as String?),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Categoría: ${paso['categoria_detectada'] ?? '-'}'),
                    Text('Ejes: ${paso['ejes_detectados'] ?? 'N/D'}'),
                    Text(
                      'Tarifa: ${(paso['monto'] as num?)?.toStringAsFixed(2) ?? '0.00'} Bs',
                      style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.greenAccent),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
              ],
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _camReady ? _toggleStream : null,
                      icon: Icon(_streaming ? Icons.stop : Icons.videocam),
                      label: Text(_streaming ? 'Detener video' : 'Video (2 fps)'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _streaming ? Colors.redAccent : Colors.teal,
                        foregroundColor: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: (_camReady && !_sending) ? _capturarManual : null,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Capturar paso'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}
