import sys
from absl import app

AVAILABLE_SCRIPTS = [
    'preprocess', 
    'update',
    'info',
    'add_embedding',
    'presets',
    'ui'
]


def help():
    print(f"""usage: acids-dataset [ {' | '.join(AVAILABLE_SCRIPTS)} ]

positional arguments:
  command     Command to launch with acids-dataset. valid commands : {' | '.join(AVAILABLE_SCRIPTS)} 
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in AVAILABLE_SCRIPTS:
        help()

    command = sys.argv[1]

    if command == 'preprocess':
        from acids_dataset import preprocess
        preprocess.import_flags()
        sys.argv[0] = preprocess.__name__
        app.run(preprocess.main)
    if command == 'update':
        from acids_dataset import update
        update.import_flags()
        sys.argv[0] = update.__name__
        app.run(update.main)
    elif command == 'info':
        from acids_dataset import info 
        info.import_flags()
        sys.argv[0] = info.__name__
        app.run(info.main)
    elif command == "ui":
        from acids_dataset.ui import cli as ui_cli
        ui_cli.import_flags()
        sys.argv[0] = ui_cli.__name__
        app.run(ui_cli.main)
    elif command == "presets":
        from acids_dataset import presets
        presets.import_flags()
        sys.argv[0] = presets.__name__
        app.run(presets.main)
    elif command == "add_embedding": 
        from acids_dataset import add_embedding
        add_embedding.import_flags()
        sys.argv[0] = add_embedding.__name__
        app.run(add_embedding.main)
        
