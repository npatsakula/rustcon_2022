# evac

- Текст доклада: [REPORT.md](./REPORT.md).

## Возможности

- [x] Базовые математические операции: сложение, вычитание, умножение и деление.
- [ ] Функции.
    - [x] Базовые тригонометрические функции: `sin`, `cos`, `pi`.
    - [x] Вложенный вызов функций из контекста.
    - [ ] JIT.
- [ ] Предупреждения.
    - [ ] Не используемая переменная.

## Окружение для разработки

### NIX

```bash
# Устанавливаем NIX:
sh <(curl -L https://nixos.org/nix/install) --no-daemon
# Создаём директорию конфигурационных файлов:
mkdir -p ~/.config/nix
# Добавляем поддержку flakes:
echo "experimental-features = flakes nix-command" >> ~/.config/nix/nix.conf 
# Открываем терминал в окружении разработки:
nix develop -j$(nproc) .
```

### Bare metal

* Профилировщик `valgrind` с утилитой `cachegrind`.
* Аллоктор `tcmalloc`.
* Компиляторный комплект LLVM 14.

## Тесты

### Unit-тесты

```bash
cargo test
```

### Тесты производительности

```bash
# Тесты на скорость парсинга и время вычисления:
cargo bench --profile bench --bench benchmark

# Профилирование:
cargo bench --profile bench --bench benchmark -- --profile-time 30
# Результаты профилирования могут быть отражены следующим образом:
# firefox ./target/criterion/{bench_name}/500000/profile/flamegraph.svg

# Профилирование доступа к памяти:
cargo bench --profile cache
```