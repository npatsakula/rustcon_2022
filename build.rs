extern crate rand;
extern crate regex_generate;

#[allow(dead_code)]
fn generate_names(count: usize) {
    // Требуется, чтобы записать код в файл.
    use std::io::Write;

    let mut generator = regex_generate::Generator::new(
        // Имя состоит из символов класса `alpha`, длина от двух до двенадцати символов.
        r"\p{alpha}{2,12}",
        rand::thread_rng(),
        regex_generate::DEFAULT_MAX_REPEAT,
    )
    .unwrap();

    // Мы не хотим, чтобы имена повторялись, потому используем HashSet:
    let mut names = std::collections::HashSet::with_capacity(count);
    while names.len() < count {
        let mut buffer = vec![];
        generator.generate(&mut buffer).unwrap();
        let name = String::from_utf8(buffer).unwrap();
        names.insert(name);
    }

    let names: Vec<_> = names.into_iter().collect();
    // Описываем структуру, которую хотим генерировать:
    let names = quote::quote! {
        pub(crate) const NAMES: [&str; #count] = [
            #(#names),*
        ];
    };

    let names = names.to_string();
    // Записываем её в файл:
    std::fs::File::create("./src/names.rs")
        .unwrap()
        .write_all(names.as_bytes())
        .unwrap();
}

fn main() {
    // generate_names(300);
    lalrpop::process_root().unwrap();
}
