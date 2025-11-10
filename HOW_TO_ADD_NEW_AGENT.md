# Как добавить новую стратегию (агента) в Streamlit интерфейс

## 📋 Общая информация

Этот документ описывает пошаговый процесс добавления нового AI-агента для игры 2048 в веб-интерфейс.

---

## 🎯 Шаг 1: Создание нового агента

### 1.1. Создайте класс агента в `agents_2048.py`

Ваш новый агент должен наследоваться от базового класса `Agent2048` и реализовать метод `choose_action`:

```python
class MyNewAgent(Agent2048):
    """
    Описание вашей стратегии
    
    Пример: Агент, который всегда пытается двигаться по диагонали
    """
    
    def __init__(self):
        super().__init__()
        self.name = "My New Agent"  # Имя для отображения в UI
    
    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """
        Выбор следующего хода на основе текущего состояния игры
        
        Args:
            game: Текущее состояние игры
            
        Returns:
            Direction или None если ходов нет
        """
        # Получите доступные ходы
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return None
        
        # Ваша логика выбора хода
        # Например, приоритет: UP > RIGHT > DOWN > LEFT
        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            if direction in available_moves:
                return direction
        
        return available_moves[0] if available_moves else None
```

### 1.2. Экспортируйте агента

Убедитесь, что ваш класс экспортируется в `__init__.py` модуля `agents_2048`:

```python
from .my_new_agent import MyNewAgent

__all__ = [
    'Agent2048',
    'RandomAgent',
    'GreedyAgent',
    'CornerAgent',
    'MonotonicAgent',
    'MyNewAgent',  # Добавьте сюда
]
```

---

## 🖥️ Шаг 2: Добавление в Streamlit интерфейс

### 2.1. Импортируйте агента в `streamlit_2048.py`

В начале файла добавьте импорт вашего агента:

```python
from agents_2048 import (
    RandomAgent,
    GreedyAgent,
    CornerAgent,
    MonotonicAgent,
    MyNewAgent,  # Добавьте сюда
)
```

### 2.2. Добавьте агента в страницу "Watch Agents"

Найдите функцию `watch_agent_page()` (примерно строка 410) и добавьте ваш агент в словарь:

```python
def watch_agent_page():
    """Watch agent play interface"""
    st.title("🤖 Watch AI Agents Play")

    # Agent selection
    agent_options = {
        "Random Agent": RandomAgent(),
        "Greedy Agent": GreedyAgent(),
        "Corner Agent (Top-Left)": CornerAgent("top-left"),
        "Corner Agent (Top-Right)": CornerAgent("top-right"),
        "Monotonic Agent": MonotonicAgent(),
        "My New Agent": MyNewAgent(),  # ← Добавьте здесь
    }
```

**Важно**: Ключ словаря - это имя, которое увидит пользователь в выпадающем списке.

### 2.3. Добавьте агента в страницу "Compare Agents"

Найдите функцию `compare_agents_page()` (примерно строка 547) и добавьте чекбокс для вашего агента:

```python
def compare_agents_page():
    # ... существующий код ...
    
    # Agent selection
    st.markdown("### Select Agents to Compare")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_random = st.checkbox("Random Agent", value=True)
        use_greedy = st.checkbox("Greedy Agent", value=True)

    with col2:
        use_corner = st.checkbox("Corner Agent", value=True)
        corner_position = st.selectbox(
            "Corner Position", ["top-left", "top-right"], disabled=not use_corner
        )

    with col3:
        use_monotonic = st.checkbox("Monotonic Agent", value=True)
        use_mynew = st.checkbox("My New Agent", value=False)  # ← Добавьте здесь
```

### 2.4. Добавьте логику добавления агента в сравнение

В той же функции `compare_agents_page()`, найдите блок формирования списка `agents_to_compare`:

```python
    if st.button("🚀 Start Comparison", width="stretch", type="primary"):
        agents_to_compare = []

        if use_random:
            agents_to_compare.append(("Random", RandomAgent()))
        if use_greedy:
            agents_to_compare.append(("Greedy", GreedyAgent()))
        if use_corner:
            agents_to_compare.append(
                (f"Corner ({corner_position})", CornerAgent(corner_position))
            )
        if use_monotonic:
            agents_to_compare.append(("Monotonic", MonotonicAgent()))
        if use_mynew:  # ← Добавьте здесь
            agents_to_compare.append(("My New Agent", MyNewAgent()))
```

---

## ✅ Шаг 3: Проверка

### 3.1. Запустите приложение

```bash
streamlit run streamlit_2048.py
```

### 3.2. Проверьте функциональность

1. **Страница "Watch Agents"**:
   - Откройте вкладку "🤖 Watch Agents"
   - Проверьте, что ваш агент появился в выпадающем списке
   - Запустите симуляцию с вашим агентом
   - Убедитесь, что игра проходит без ошибок

2. **Страница "Compare Agents"**:
   - Откройте вкладку "⚔️ Compare Agents"
   - Проверьте, что чекбокс для вашего агента отображается
   - Включите ваш агент и несколько других
   - Запустите сравнение
   - Проверьте, что результаты отображаются корректно

3. **Страница "Analytics"**:
   - После запуска симуляций проверьте вкладку "📊 Analytics"
   - Убедитесь, что данные вашего агента отображаются в графиках

---

## 🎨 Шаг 4 (Опционально): Настройка UI

### 4.1. Если у вашего агента есть параметры

Если ваш агент имеет настраиваемые параметры (как `CornerAgent` с позицией угла), добавьте дополнительные контролы:

```python
with col3:
    use_mynew = st.checkbox("My New Agent", value=False)
    
    # Дополнительные параметры для вашего агента
    mynew_param = st.selectbox(
        "Strategy Type", 
        ["aggressive", "defensive"], 
        disabled=not use_mynew
    )
```

И используйте параметр при создании агента:

```python
if use_mynew:
    agents_to_compare.append(
        (f"My New Agent ({mynew_param})", MyNewAgent(strategy=mynew_param))
    )
```

### 4.2. Изменение порядка в списке

Агенты отображаются в том порядке, в котором они добавлены в словарь `agent_options`. Чтобы изменить порядок, просто переставьте строки в словаре.

### 4.3. Установка агента по умолчанию

В функции `watch_agent_page()` найдите строку с `st.selectbox`:

```python
selected_agent_name = st.selectbox(
    "Select Agent", 
    list(agent_options.keys()), 
    index=4  # ← Измените индекс на позицию вашего агента (начиная с 0)
)
```

---

## 📝 Чек-лист для добавления агента

- [ ] Создан класс агента в `agents_2048.py`
- [ ] Класс наследуется от `Agent2048`
- [ ] Реализован метод `choose_action`
- [ ] Установлено свойство `self.name`
- [ ] Агент добавлен в импорты `streamlit_2048.py`
- [ ] Агент добавлен в словарь `agent_options` в `watch_agent_page()`
- [ ] Добавлен чекбокс в `compare_agents_page()`
- [ ] Добавлена логика в `agents_to_compare`
- [ ] Протестирована страница "Watch Agents"
- [ ] Протестирована страница "Compare Agents"
- [ ] Проверена страница "Analytics"

---

## 🐛 Частые ошибки и их решения

### Ошибка: "NameError: name 'MyNewAgent' is not defined"

**Причина**: Агент не импортирован в `streamlit_2048.py`

**Решение**: Добавьте импорт в начало файла:
```python
from agents_2048 import (..., MyNewAgent)
```

### Ошибка: "KeyError: 'total_moves'"

**Причина**: Метод `play_game()` возвращает не все ожидаемые ключи

**Решение**: Убедитесь, что вы используете `game.moves_count` напрямую, а не из результата:
```python
result = agent.play_game(game)
moves.append(game.moves_count)  # Правильно
# НЕ: moves.append(result["total_moves"])  # Неправильно
```

### Агент не появляется в списке

**Причина**: Не добавлен в словарь `agent_options`

**Решение**: Проверьте, что вы добавили агент в словарь в функции `watch_agent_page()`

### Сравнение не работает

**Причина**: Не добавлена логика в блок формирования `agents_to_compare`

**Решение**: Добавьте проверку чекбокса и создание экземпляра агента в `compare_agents_page()`

---

## 💡 Советы по разработке агентов

1. **Начните с простого**: Протестируйте базовую версию агента перед добавлением сложной логики

2. **Используйте существующие агенты как примеры**: Посмотрите на реализацию `GreedyAgent` или `MonotonicAgent`

3. **Тестируйте локально**: Прежде чем добавлять в UI, протестируйте агент отдельно:
   ```python
   from game2048_engine import Game2048
   from agents_2048 import MyNewAgent
   
   agent = MyNewAgent()
   game = Game2048(seed=42)
   result = agent.play_game(game, verbose=True)
   print(result)
   ```

4. **Документируйте стратегию**: Добавьте docstring с описанием логики вашего агента

5. **Проверьте производительность**: Если агент работает медленно, это будет заметно в UI при визуализации

---

## 📚 Дополнительные ресурсы

- Документация базового класса `Agent2048`: см. `api_info.md`
- Примеры существующих агентов: см. файл `agents_2048.py`
- API игрового движка: см. `game2048_engine.py`

---

## 🎓 Пример: Добавление "Smart Diagonal Agent"

Полный пример добавления агента, который предпочитает диагональные движения:

### agents_2048.py
```python
class SmartDiagonalAgent(Agent2048):
    """Agent that prefers diagonal-like movements (alternating UP-RIGHT or DOWN-LEFT)"""
    
    def __init__(self, preference="up-right"):
        super().__init__()
        self.name = f"Smart Diagonal ({preference})"
        self.preference = preference
        
    def choose_action(self, game: Game2048) -> Optional[Direction]:
        available_moves = game.get_available_moves()
        if not available_moves:
            return None
        
        if self.preference == "up-right":
            priority = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        else:  # down-left
            priority = [Direction.DOWN, Direction.LEFT, Direction.UP, Direction.RIGHT]
        
        for direction in priority:
            if direction in available_moves:
                return direction
        
        return available_moves[0]
```

### streamlit_2048.py

**Импорт:**
```python
from agents_2048 import (
    ...,
    SmartDiagonalAgent,
)
```

**watch_agent_page():**
```python
agent_options = {
    ...,
    "Smart Diagonal (Up-Right)": SmartDiagonalAgent("up-right"),
    "Smart Diagonal (Down-Left)": SmartDiagonalAgent("down-left"),
}
```

**compare_agents_page():**
```python
# В секции чекбоксов:
with col3:
    ...
    use_diagonal = st.checkbox("Smart Diagonal Agent", value=False)
    diagonal_pref = st.selectbox(
        "Diagonal Preference",
        ["up-right", "down-left"],
        disabled=not use_diagonal
    )

# В секции формирования списка:
if use_diagonal:
    agents_to_compare.append(
        (f"Smart Diagonal ({diagonal_pref})", SmartDiagonalAgent(diagonal_pref))
    )
```

---

**Удачи в разработке новых стратегий! 🚀**
