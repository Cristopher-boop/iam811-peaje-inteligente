import 'package:flutter_test/flutter_test.dart';
import 'package:peaje_movil/api_client.dart';
import 'package:peaje_movil/config.dart';
import 'package:peaje_movil/main.dart';

void main() {
  testWidgets('Renderiza pantalla de login Google', (WidgetTester tester) async {
    final config = AppConfig(
      backendUrl: 'https://peaje-iam811.onrender.com',
      googleServerClientId: '',
    );
    await tester.pumpWidget(PeajeApp(api: TollApiClient(config)));
    expect(find.text('Control Inteligente de Peaje'), findsOneWidget);
    expect(find.text('Iniciar sesión con Google'), findsOneWidget);
  });
}
