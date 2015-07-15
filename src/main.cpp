#include <QApplication>
#include <QTextCodec>
#include <QTranslator>

#include <cstdio>
#include <unistd.h>

#include <QMessageBox>
//#include "tst/test_intervalshistogram.h"

#include "inc/mainwindow.h"

uint nbT = 1;

void outputHandler(QtMsgType type, const char *msg)
{
	fprintf(stderr, "%s\n", msg);
	switch (type) {
		case QtWarningMsg:
			QMessageBox::warning(0,QObject::tr("Message d'erreur"),QObject::tr(msg));
			break;
		default:
			break;
	}
}

int main(int argc, char *argv[])
{
//	qInstallMsgHandler(outputHandler);
  int opt;
  
  while ((opt = getopt(argc, argv, "t:")) != -1) {
    switch (opt) {
    case 't':
      nbT = atoi(optarg);
      break;
    }
  }

#ifdef PARALLEL
  std::cout << nbT << " processus actif(s) / " << omp_get_num_procs() << std::endl;
#endif

	QTextCodec::setCodecForTr(QTextCodec::codecForName("UTF-8"));
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));

	QApplication app(argc, argv);

	QTranslator translator;
	translator.load("TKDetection_en");
	app.installTranslator(&translator);

	MainWindow w;
	w.show();
	return app.exec();

//	Test_IntervalsHistogram::allTests();
//	return 0;
}
