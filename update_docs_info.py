import inspect
import importlib
import sys
from pathlib import Path

# Добавляем корневую папку проекта и src в sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def extract_api(module_name):
    """Извлекает только сигнатуры и докстринги"""
    module = importlib.import_module(module_name)

    output = [f"# Module: {module_name}\n"]

    if module.__doc__:
        output.append(f"{module.__doc__}\n")

    # Классы
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module_name:  # Только из этого модуля
            output.append(f"\n## Class: {name}")
            output.append(f"```python\nclass {name}:\n```")
            if obj.__doc__:
                output.append(f"{inspect.cleandoc(obj.__doc__)}\n")

            # Методы класса
            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                if not method_name.startswith("_"):  # Пропускаем приватные
                    sig = inspect.signature(method)
                    output.append(f"### {name}.{method_name}{sig}")
                    if method.__doc__:
                        output.append(f"{inspect.cleandoc(method.__doc__)}\n")

    # Функции
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module_name:
            sig = inspect.signature(obj)
            output.append(f"\n## Function: {name}{sig}")
            if obj.__doc__:
                output.append(f"{inspect.cleandoc(obj.__doc__)}\n")

    return "\n".join(output)


if __name__ == "__main__":
    modules_list = [
        "game2048_engine",
        "agents_2048",
        "terminal_2048",
        "streamlit_2048",
    ]

    for module_name in modules_list:
        print(extract_api(module_name))
        print("\n" + "=" * 80 + "\n")

    output_path = ".docs/api_info.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# API Documentation\n\n")
        f.write(f"Generated for modules: {', '.join(modules_list)}\n\n")
        f.write("=" * 80 + "\n\n")

        for module_name in modules_list:
            print(f"Processing {module_name}...", file=sys.stderr)
            content = extract_api(module_name)
            f.write(content)
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"✓ Документация сохранена в {output_path}", file=sys.stderr)
