# evac

- Текст доклада: [REPORT.md](./REPORT.md).

## Возможности

```rust
fn double(a, b) = a * b;

fn triple(a, b, c) = double(a, b) * c;

triple(1, 2 + 2, sin(3 + 3))
```

- [x] Базовые математические операции: сложение, вычитание, умножение и деление.
- [x] Функции.
    - [x] Базовые тригонометрические функции: `sin`, `cos`, `pi`.
    - [x] Вложенный вызов функций из контекста.
    - [x] JIT.

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

* Компиляторный комплект Rust.
* Компиляторный комплект LLVM 14.
* Библиотеки: `libffi`, `zlib`, `libxml2`, `libncurses`.

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
```

## Docker контейнер

Мы можем собрать и загрузить docker-контейнер следующим образом:

```bash
nix build '.#container'
docker load < result
```