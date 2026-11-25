import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'api_client.dart';

class _YuvPayload {
  final Uint8List y;
  final Uint8List u;
  final Uint8List v;
  final int width;
  final int height;
  final int yRowStride;
  final int uRowStride;
  final int vRowStride;
  final int yPixelStride;
  final int uPixelStride;
  final int vPixelStride;
  final int rotationDegrees;

  const _YuvPayload({
    required this.y,
    required this.u,
    required this.v,
    required this.width,
    required this.height,
    required this.yRowStride,
    required this.uRowStride,
    required this.vRowStride,
    required this.yPixelStride,
    required this.uPixelStride,
    required this.vPixelStride,
    required this.rotationDegrees,
  });
}

Uint8List _encodeYuvToJpeg(_YuvPayload p) {
  final step = (p.width > 960 || p.height > 960) ? 2 : 1;
  final outW = (p.width / step).ceil();
  final outH = (p.height / step).ceil();
  final out = img.Image(width: outW, height: outH);

  for (int oy = 0, y = 0; oy < outH && y < p.height; oy++, y += step) {
    final uvRow = y >> 1;
    final yRowOffset = y * p.yRowStride;
    final uRowOffset = uvRow * p.uRowStride;
    final vRowOffset = uvRow * p.vRowStride;

    for (int ox = 0, x = 0; ox < outW && x < p.width; ox++, x += step) {
      final uvCol = x >> 1;
      final yIdx = yRowOffset + x * p.yPixelStride;
      final uIdx = uRowOffset + uvCol * p.uPixelStride;
      final vIdx = vRowOffset + uvCol * p.vPixelStride;
      if (yIdx >= p.y.length || uIdx >= p.u.length || vIdx >= p.v.length) continue;

      final yp = p.y[yIdx];
      final up = p.u[uIdx] - 128;
      final vp = p.v[vIdx] - 128;

      int r = yp + ((vp * 1436) >> 10);
      int g = yp - ((up * 352 + vp * 731) >> 10);
      int b = yp + ((up * 1814) >> 10);

      if (r < 0) r = 0;
      if (r > 255) r = 255;
      if (g < 0) g = 0;
      if (g > 255) g = 255;
      if (b < 0) b = 0;
      if (b > 255) b = 255;

      out.setPixelRgba(ox, oy, r, g, b, 255);
    }
  }

  final oriented = p.rotationDegrees == 0 ? out : img.copyRotate(out, angle: p.rotationDegrees);
  final maxDim = oriented.width > oriented.height ? oriented.width : oriented.height;
  final resized = maxDim > 720
      ? (oriented.width >= oriented.height
          ? img.copyResize(oriented, width: 720)
          : img.copyResize(oriented, height: 720))
      : oriented;

  return Uint8List.fromList(img.encodeJpg(resized, quality: 78));
}

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
  Map<String, dynamic>? _lastInference;
  int _framesOk = 0;
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
      final trasera = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );
      _cam = CameraController(
        trasera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _cam!.initialize();
      if (!mounted) return;
      setState(() {
        _camReady = true;
        _status = 'Listo · Apunta al video y pulsa Video (2 fps) o Capturar';
      });
    } catch (e) {
      if (mounted) setState(() => _status = 'Error cámara: $e');
    }
  }

  @override
  void dispose() {
    final cam = _cam;
    _cam = null;
    if (cam != null) {
      if (cam.value.isStreamingImages) {
        unawaited(cam.stopImageStream().then((_) => cam.dispose()));
      } else {
        cam.dispose();
      }
    }
    super.dispose();
  }

  int _computeRotationDegrees() {
    if (_cam == null) return 0;
    final desc = _cam!.description;
    final devDeg = switch (_cam!.value.deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };
    if (desc.lensDirection == CameraLensDirection.front) {
      return (desc.sensorOrientation + devDeg) % 360;
    }
    return (desc.sensorOrientation - devDeg + 360) % 360;
  }

  Future<void> _toggleStream() async {
    if (!_camReady || _cam == null) return;
    if (_streaming) {
      await _stopStream();
      return;
    }

    setState(() {
      _streaming = true;
      _status = 'Video continuo activo (2 fps) · Analizando...';
    });

    await _cam!.startImageStream((CameraImage image) async {
      if (!_streaming || _sending || !mounted) return;
      final now = DateTime.now();
      if (now.difference(_lastSent).inMilliseconds < 450) return;
      _lastSent = now;
      _sending = true;
      try {
        final bytes = await _encodeCameraImage(image);
        if (bytes != null && mounted && _streaming) {
          await _procesarBytes(bytes, modo: 'stream');
        }
      } finally {
        _sending = false;
      }
    });
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
    if (_streaming) await _stopStream();
    setState(() {
      _sending = true;
      _status = 'Capturando foto y analizando en RunPod...';
    });
    try {
      final shot = await _cam!.takePicture();
      final raw = await shot.readAsBytes();
      await _procesarBytes(raw, modo: 'manual');
    } catch (e) {
      if (mounted) setState(() => _status = 'Error: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _procesarBytes(Uint8List jpeg, {required String modo}) async {
    try {
      final res = await widget.api.sendFrame(jpeg, widget.casetaId, modo: modo);
      if (!mounted) return;
      final paso = res['paso'] as Map<String, dynamic>?;
      final inf = res['inferencia'] as Map<String, dynamic>?;
      final lat = (inf?['latencia_ms']?['total']) ?? paso?['latencia_ms'] ?? '-';
      final motivo = res['motivo'] as String?;

      setState(() {
        _framesOk++;
        if (inf != null) _lastInference = inf;
        if (paso != null) {
          _lastResult = res;
          final placaTxt = (paso['placa_normalizada'] as String?)?.isNotEmpty == true
              ? paso['placa_normalizada'] as String
              : (paso['placa_leida'] ?? 'SIN PLACA').toString();
          final tag = motivo == 'cooldown_placa' || motivo == 'cooldown_vehiculo'
              ? 'Ya registrado (cooldown)'
              : 'Registrado';
          _status = '$tag · $placaTxt · ${paso['categoria_detectada'] ?? '-'} · $lat ms';
        } else {
          final cat = inf?['vehiculo']?['categoria'];
          _status = cat != null
              ? 'Detectado $cat (leyendo placa...) · $lat ms'
              : 'Video activo (#$_framesOk) · Sin vehículo · $lat ms';
        }
      });
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Error API: ${e.toString().replaceFirst('Exception: ', '')}');
      }
    }
  }

  Future<Uint8List?> _encodeCameraImage(CameraImage image) async {
    try {
      if (image.planes.length < 2) return null;
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes.length > 2 ? image.planes[2] : image.planes[1];

      final payload = _YuvPayload(
        y: Uint8List.fromList(yPlane.bytes),
        u: Uint8List.fromList(uPlane.bytes),
        v: Uint8List.fromList(vPlane.bytes),
        width: image.width,
        height: image.height,
        yRowStride: yPlane.bytesPerRow,
        uRowStride: uPlane.bytesPerRow,
        vRowStride: vPlane.bytesPerRow,
        yPixelStride: yPlane.bytesPerPixel ?? 1,
        uPixelStride: uPlane.bytesPerPixel ?? 1,
        vPixelStride: vPlane.bytesPerPixel ?? 1,
        rotationDegrees: _computeRotationDegrees(),
      );
      return await compute(_encodeYuvToJpeg, payload);
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
      case 'ILEGIBLE':
        return Colors.orangeAccent;
      default:
        return Colors.white70;
    }
  }

  Widget _buildCameraPane() {
    return Stack(
      fit: StackFit.expand,
      children: [
        Container(
          color: Colors.black,
          child: (_camReady && _cam != null)
              ? Center(child: CameraPreview(_cam!))
              : Center(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(_status, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white70)),
                  ),
                ),
        ),
        Positioned(
          top: 8,
          left: 8,
          right: 8,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.7),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: _streaming ? Colors.tealAccent.withValues(alpha: 0.7) : Colors.white24,
              ),
            ),
            child: Row(
              children: [
                Icon(
                  _streaming ? Icons.fiber_manual_record : Icons.videocam_outlined,
                  size: 14,
                  color: _streaming ? Colors.redAccent : Colors.white70,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    _status,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 12, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildControlsPane({required bool isLandscape}) {
    final paso = _lastResult?['paso'] as Map<String, dynamic>?;
    final catLive = paso?['categoria_detectada'] ?? _lastInference?['vehiculo']?['categoria'] ?? '-';
    final ejesLive = paso?['ejes_detectados'] ?? _lastInference?['ejes'];
    final latLive = (_lastInference?['latencia_ms']?['total']) ?? paso?['latencia_ms'];
    final saldoRestante = _lastResult?['saldo_restante'];
    final propietario = _lastResult?['propietario'];

    final content = Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (paso != null) ...[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                child: Text(
                  (paso['placa_normalizada'] as String?)?.isNotEmpty == true
                      ? paso['placa_normalizada'] as String
                      : (paso['placa_leida'] ?? 'SIN PLACA').toString(),
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: isLandscape ? 20 : 22,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.4,
                  ),
                ),
              ),
              const SizedBox(width: 8),
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
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: _estadoColor(paso['estado'] as String?),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Wrap(
            spacing: 12,
            runSpacing: 4,
            alignment: WrapAlignment.spaceBetween,
            children: [
              Text('Categoría: $catLive', style: const TextStyle(fontSize: 13)),
              Text('Ejes: ${ejesLive ?? 'N/D'}', style: const TextStyle(fontSize: 13)),
              Text(
                'Tarifa: ${(paso['monto'] as num?)?.toStringAsFixed(2) ?? '0.00'} Bs',
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Colors.greenAccent),
              ),
            ],
          ),
          if (propietario != null || saldoRestante != null || latLive != null) ...[
            const SizedBox(height: 4),
            Text(
              [
                if (propietario != null) propietario.toString(),
                if (saldoRestante != null) 'Saldo: ${(saldoRestante as num).toStringAsFixed(2)} Bs',
                if (latLive != null) 'GPU: $latLive ms',
              ].join(' · '),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 11, color: Colors.white60),
            ),
          ],
          const SizedBox(height: 10),
        ] else ...[
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 6),
            child: Text(
              'Enfoca un vehículo o la pantalla de video y presiona Video (2 fps) para detección automática.',
              style: TextStyle(fontSize: 12, color: Colors.white70),
            ),
          ),
        ],
        Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: _camReady ? _toggleStream : null,
                icon: Icon(_streaming ? Icons.stop : Icons.videocam),
                label: Text(_streaming ? 'Detener' : 'Video (2 fps)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _streaming ? Colors.redAccent : Colors.teal,
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: isLandscape ? 10 : 12),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: (_camReady && !_sending) ? _capturarManual : null,
                icon: const Icon(Icons.camera_alt),
                label: const Text('Capturar paso'),
                style: OutlinedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: isLandscape ? 10 : 12),
                ),
              ),
            ),
          ],
        ),
      ],
    );

    return Container(
      padding: EdgeInsets.all(isLandscape ? 12 : 14),
      color: const Color(0xFF0F172A),
      child: isLandscape ? SingleChildScrollView(child: content) : content,
    );
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isLandscape = constraints.maxWidth > constraints.maxHeight;
        if (isLandscape) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(flex: 6, child: _buildCameraPane()),
              Expanded(flex: 4, child: _buildControlsPane(isLandscape: true)),
            ],
          );
        }
        return Column(
          children: [
            Expanded(child: _buildCameraPane()),
            _buildControlsPane(isLandscape: false),
          ],
        );
      },
    );
  }
}
