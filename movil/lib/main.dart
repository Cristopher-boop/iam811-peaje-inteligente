import 'package:flutter/material.dart';
import 'api_client.dart';
import 'caseta_page.dart';
import 'config.dart';
import 'consulta_page.dart';
import 'login_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final config = await AppConfig.load();
  final api = TollApiClient(config);
  runApp(PeajeApp(api: api));
}

class PeajeApp extends StatefulWidget {
  final TollApiClient api;

  const PeajeApp({super.key, required this.api});

  @override
  State<PeajeApp> createState() => _PeajeAppState();
}

class _PeajeAppState extends State<PeajeApp> {
  AuthSession? _session;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Peaje AI — IAM-811',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal, brightness: Brightness.dark),
      ),
      home: _session == null
          ? LoginPage(
              api: widget.api,
              onAuthenticated: (s) => setState(() => _session = s),
            )
          : HomeShell(
              api: widget.api,
              session: _session!,
              onLogout: () {
                widget.api.session = null;
                setState(() => _session = null);
              },
            ),
    );
  }
}

class HomeShell extends StatefulWidget {
  final TollApiClient api;
  final AuthSession session;
  final VoidCallback onLogout;

  const HomeShell({
    super.key,
    required this.api,
    required this.session,
    required this.onLogout,
  });

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;
  late int _casetaId;

  @override
  void initState() {
    super.initState();
    _casetaId = widget.session.casetas.isNotEmpty
        ? (widget.session.casetas.first['id'] as int)
        : 1;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.session.nombre, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
            Text('${widget.session.rol.toUpperCase()} · ${widget.session.email}', style: const TextStyle(fontSize: 11, color: Colors.white70)),
          ],
        ),
        actions: [
          if (widget.session.casetas.isNotEmpty)
            DropdownButton<int>(
              value: _casetaId,
              underline: const SizedBox.shrink(),
              items: widget.session.casetas
                  .map((c) => DropdownMenuItem<int>(
                        value: c['id'] as int,
                        child: Text(c['nombre'] as String, style: const TextStyle(fontSize: 12)),
                      ))
                  .toList(),
              onChanged: (v) {
                if (v != null) setState(() => _casetaId = v);
              },
            ),
          IconButton(icon: const Icon(Icons.logout), onPressed: widget.onLogout),
        ],
      ),
      body: _tab == 0
          ? CasetaPage(api: widget.api, casetaId: _casetaId)
          : ConsultaPage(api: widget.api),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.videocam), label: 'Modo Caseta'),
          NavigationDestination(icon: Icon(Icons.chat), label: 'Modo Consulta'),
        ],
      ),
    );
  }
}
