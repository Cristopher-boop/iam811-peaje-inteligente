import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'config.dart';

class AuthSession {
  final String jwt;
  final String email;
  final String nombre;
  final String rol;
  final List<Map<String, dynamic>> casetas;

  AuthSession({
    required this.jwt,
    required this.email,
    required this.nombre,
    required this.rol,
    required this.casetas,
  });
}

class TollApiClient {
  final AppConfig config;
  AuthSession? session;

  TollApiClient(this.config);

  Map<String, String> _authHeaders([Map<String, String>? extra]) {
    final h = <String, String>{};
    if (session != null) {
      h['Authorization'] = 'Bearer ${session!.jwt}';
    }
    if (extra != null) h.addAll(extra);
    return h;
  }

  Future<AuthSession> loginWithGoogleToken(String idToken) async {
    final uri = Uri.parse('${config.backendUrl}/api/v1/auth/google');
    final resp = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'id_token': idToken}),
        )
        .timeout(const Duration(seconds: 65));

    final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    if (resp.statusCode != 200) {
      throw Exception(data['detail'] ?? 'Error de autenticacion (${resp.statusCode})');
    }

    final u = data['usuario'] as Map<String, dynamic>;
    final rawCasetas = (data['casetas'] as List<dynamic>? ?? []);
    session = AuthSession(
      jwt: data['access_token'] as String,
      email: u['email_google'] as String,
      nombre: u['nombre'] as String,
      rol: u['rol'] as String,
      casetas: rawCasetas.map((e) => Map<String, dynamic>.from(e as Map)).toList(),
    );
    return session!;
  }

  Future<Map<String, dynamic>> sendFrame(
    Uint8List jpegBytes,
    int casetaId, {
    String modo = 'manual',
  }) async {
    final uri = Uri.parse('${config.backendUrl}/api/v1/pasos/frame');
    final req = http.MultipartRequest('POST', uri)
      ..headers.addAll(_authHeaders())
      ..fields['caseta_id'] = casetaId.toString()
      ..fields['modo'] = modo
      ..files.add(http.MultipartFile.fromBytes('file', jpegBytes, filename: 'frame.jpg'));

    final streamed = await req.send().timeout(const Duration(seconds: 40));
    final resp = await http.Response.fromStream(streamed);
    final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    if (resp.statusCode != 200) {
      throw Exception(data['detail'] ?? 'Error (${resp.statusCode})');
    }
    return data;
  }

  Future<Map<String, dynamic>> queryAgent({String mensaje = '', Uint8List? imageBytes}) async {
    final uri = Uri.parse('${config.backendUrl}/api/v1/agente/chat');
    if (imageBytes != null) {
      final req = http.MultipartRequest('POST', uri)
        ..headers.addAll(_authHeaders())
        ..fields['mensaje'] = mensaje
        ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: 'placa.jpg'));
      final streamed = await req.send().timeout(const Duration(seconds: 30));
      final resp = await http.Response.fromStream(streamed);
      final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
      if (resp.statusCode != 200) {
        throw Exception(data['detail'] ?? 'Error (${resp.statusCode})');
      }
      return data;
    }

    final resp = await http
        .post(
          uri,
          headers: _authHeaders({'Content-Type': 'application/json'}),
          body: jsonEncode({'mensaje': mensaje}),
        )
        .timeout(const Duration(seconds: 25));
    final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    if (resp.statusCode != 200) {
      throw Exception(data['detail'] ?? 'Error (${resp.statusCode})');
    }
    return data;
  }

  Future<Map<String, dynamic>> fetchResumenCaseta(int casetaId) async {
    final uri = Uri.parse('${config.backendUrl}/api/v1/casetas/$casetaId/resumen');
    final resp = await http.get(uri, headers: _authHeaders()).timeout(const Duration(seconds: 10));
    final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    if (resp.statusCode != 200) {
      throw Exception(data['detail'] ?? 'Error (${resp.statusCode})');
    }
    return data;
  }
}
