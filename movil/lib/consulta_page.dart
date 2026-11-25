import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'api_client.dart';

class _ChatMsg {
  final bool isUser;
  final String text;
  final String? estado;
  final Uint8List? imageBytes;

  _ChatMsg({required this.isUser, required this.text, this.estado, this.imageBytes});
}

class ConsultaPage extends StatefulWidget {
  final TollApiClient api;

  const ConsultaPage({super.key, required this.api});

  @override
  State<ConsultaPage> createState() => _ConsultaPageState();
}

class _ConsultaPageState extends State<ConsultaPage> {
  final _ctrl = TextEditingController();
  final _picker = ImagePicker();
  final List<_ChatMsg> _msgs = [
    _ChatMsg(
      isUser: false,
      text: 'Escribe una placa (ej. 2481-KLM) o envía una foto de placa para consultar al agente.',
    ),
  ];
  bool _sending = false;

  Future<void> _enviarTexto() async {
    final txt = _ctrl.text.trim();
    if (txt.isEmpty || _sending) return;
    _ctrl.clear();
    await _enviar(mensaje: txt);
  }

  Future<void> _enviarFoto(ImageSource source) async {
    if (_sending) return;
    final xf = await _picker.pickImage(source: source, imageQuality: 85, maxWidth: 1280);
    if (xf == null) return;
    final bytes = await xf.readAsBytes();
    await _enviar(mensaje: _ctrl.text.trim(), imageBytes: bytes);
    _ctrl.clear();
  }

  Future<void> _enviar({String mensaje = '', Uint8List? imageBytes}) async {
    setState(() {
      _msgs.add(_ChatMsg(
        isUser: true,
        text: mensaje.isNotEmpty ? mensaje : '(Foto de placa)',
        imageBytes: imageBytes,
      ));
      _sending = true;
    });

    try {
      final res = await widget.api.queryAgent(mensaje: mensaje, imageBytes: imageBytes);
      setState(() {
        _msgs.add(_ChatMsg(
          isUser: false,
          text: (res['respuesta'] ?? 'Sin respuesta').toString(),
          estado: res['estado_sistema'] as String?,
        ));
      });
    } catch (e) {
      setState(() {
        _msgs.add(_ChatMsg(isUser: false, text: 'Error: ${e.toString().replaceFirst('Exception: ', '')}'));
      });
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape = MediaQuery.of(context).size.width > MediaQuery.of(context).size.height;
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: _msgs.length,
            itemBuilder: (_, i) {
              final m = _msgs[i];
              return Align(
                alignment: m.isUser ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  margin: const EdgeInsets.symmetric(vertical: 5),
                  padding: const EdgeInsets.all(12),
                  constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.82),
                  decoration: BoxDecoration(
                    color: m.isUser ? const Color(0xFF065F46) : const Color(0xFF1E293B),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (m.estado != null) ...[
                        Text(
                          'Estado: ${m.estado}',
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            color: Colors.greenAccent,
                          ),
                        ),
                        const SizedBox(height: 4),
                      ],
                      if (m.imageBytes != null) ...[
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.memory(
                            m.imageBytes!,
                            height: isLandscape ? 92 : 140,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(height: 6),
                      ],
                      Text(m.text.replaceAll('**', '')),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
        if (_sending) const LinearProgressIndicator(minHeight: 2),
        Padding(
          padding: const EdgeInsets.all(8),
          child: Row(
            children: [
              IconButton(
                tooltip: 'Tomar foto',
                icon: const Icon(Icons.camera_alt),
                onPressed: _sending ? null : () => _enviarFoto(ImageSource.camera),
              ),
              IconButton(
                tooltip: 'Galería',
                icon: const Icon(Icons.photo),
                onPressed: _sending ? null : () => _enviarFoto(ImageSource.gallery),
              ),
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  onSubmitted: (_) => _enviarTexto(),
                  decoration: const InputDecoration(
                    hintText: 'Placa o consulta...',
                    isDense: true,
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 6),
              IconButton(
                icon: const Icon(Icons.send, color: Colors.greenAccent),
                onPressed: _sending ? null : _enviarTexto,
              ),
            ],
          ),
        ),
      ],
    );
  }
}
