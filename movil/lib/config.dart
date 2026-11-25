import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  static const String defaultBackendUrl = String.fromEnvironment(
    'BACKEND_URL',
    defaultValue: 'https://peaje-iam811.onrender.com',
  );

  static const String defaultServerClientId = String.fromEnvironment(
    'GOOGLE_SERVER_CLIENT_ID',
    defaultValue: '',
  );

  String backendUrl;
  String googleServerClientId;

  AppConfig({
    required this.backendUrl,
    required this.googleServerClientId,
  });

  static Future<AppConfig> load() async {
    final prefs = await SharedPreferences.getInstance();
    final url = (prefs.getString('backend_url') ?? defaultBackendUrl).trim();
    final cid = (prefs.getString('google_server_client_id') ?? defaultServerClientId).trim();
    return AppConfig(
      backendUrl: url.endsWith('/') ? url.substring(0, url.length - 1) : url,
      googleServerClientId: cid,
    );
  }

  Future<void> save(String newUrl, String newClientId) async {
    final prefs = await SharedPreferences.getInstance();
    var cleanUrl = newUrl.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    }
    backendUrl = cleanUrl;
    googleServerClientId = newClientId.trim();
    await prefs.setString('backend_url', backendUrl);
    await prefs.setString('google_server_client_id', googleServerClientId);
  }
}
