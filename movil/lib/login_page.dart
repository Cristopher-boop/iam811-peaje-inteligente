import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'api_client.dart';

class LoginPage extends StatefulWidget {
  final TollApiClient api;
  final void Function(AuthSession session) onAuthenticated;

  const LoginPage({super.key, required this.api, required this.onAuthenticated});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  bool _loading = false;
  String? _error;

  Future<void> _signInWithGoogle() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final serverCid = widget.api.config.googleServerClientId;
      final gs = GoogleSignIn(
        scopes: const ['email', 'profile'],
        serverClientId: serverCid.isEmpty ? null : serverCid,
      );
      await gs.signOut();
      final account = await gs.signIn();
      if (account == null) {
        setState(() => _loading = false);
        return;
      }
      final auth = await account.authentication;
      final idToken = auth.idToken;
      if (idToken == null || idToken.isEmpty) {
        throw Exception(
          'Google no devolvio idToken. Configura el Web Client ID (serverClientId) en Ajustes.',
        );
      }
      final session = await widget.api.loginWithGoogleToken(idToken);
      if (mounted) widget.onAuthenticated(session);
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _openConfigDialog() {
    final urlCtrl = TextEditingController(text: widget.api.config.backendUrl);
    final cidCtrl = TextEditingController(text: widget.api.config.googleServerClientId);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Configuración del Servidor'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: urlCtrl,
              decoration: const InputDecoration(
                labelText: 'URL HTTPS del Web/API',
                hintText: 'https://peaje-iam811.onrender.com',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: cidCtrl,
              decoration: const InputDecoration(
                labelText: 'Google Web Client ID (serverClientId)',
                hintText: 'xxxx.apps.googleusercontent.com',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancelar')),
          ElevatedButton(
            onPressed: () async {
              await widget.api.config.save(urlCtrl.text, cidCtrl.text);
              if (ctx.mounted) Navigator.pop(ctx);
              setState(() {});
            },
            child: const Text('Guardar'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Peaje AI — Acceso'),
        actions: [
          IconButton(icon: const Icon(Icons.settings), onPressed: _openConfigDialog),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.toll, size: 64, color: Colors.greenAccent),
                  const SizedBox(height: 12),
                  const Text(
                    'Control Inteligente de Peaje',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Servidor: ${widget.api.config.backendUrl}',
                    style: const TextStyle(fontSize: 12, color: Colors.white60),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 24),
                  if (_error != null) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.red.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.redAccent),
                      ),
                      child: Text(
                        _error!,
                        style: const TextStyle(color: Colors.redAccent, fontSize: 13),
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _loading ? null : _signInWithGoogle,
                      icon: _loading
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.login),
                      label: Text(_loading ? 'Verificando con Google...' : 'Iniciar sesión con Google'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.black87,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Solo cuentas autorizadas en la lista blanca pueden ingresar.',
                    style: TextStyle(fontSize: 11, color: Colors.white54),
                    textAlign: TextAlign.center,
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
