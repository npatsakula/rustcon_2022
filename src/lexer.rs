use logos::{Lexer, Logos, SpannedIter};

#[inline(always)]
fn float_parser<'input>(lex: &mut Lexer<'input, Token<'input>>) -> f64 {
    fast_float::parse::<f64, &str>(lex.slice()).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, logos::Logos, derive_more::Display)]
pub enum Token<'input> {
    #[regex(r"-?\d+(\.\d*)?", float_parser)]
    #[display(fmt = "{_0}")]
    Float(f64),

    #[regex(r"\p{alpha}+")]
    #[display(fmt = "{_0}")]
    Name(&'input str),

    #[display(fmt = "fn")]
    #[token("fn")]
    Fn,

    #[token("+")]
    #[display(fmt = "+")]
    Add,

    #[token("-")]
    #[display(fmt = "-")]
    Subtract,

    #[token("*")]
    #[display(fmt = "*")]
    Product,

    #[token("/")]
    #[display(fmt = "/")]
    Divine,

    #[token("(")]
    #[display(fmt = "(")]
    OpenBracket,

    #[token(")")]
    #[display(fmt = ")")]
    CloseBracket,

    #[token(",")]
    #[display(fmt = ",")]
    Comma,

    #[token("=")]
    #[display(fmt = "=")]
    Eq,

    #[token(";")]
    #[display(fmt = ";")]
    Semicolon,

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    #[display(fmt = "")]
    Error,
}

pub struct EvacLexer<'input> {
    lexer: SpannedIter<'input, Token<'input>>,
}

impl<'input> EvacLexer<'input> {
    pub fn new(source: &'input str) -> Self {
        Self {
            lexer: Token::lexer(source).spanned(),
        }
    }
}

impl<'input> Iterator for EvacLexer<'input> {
    type Item = (usize, Token<'input>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.lexer.next().map(|(t, r)| (r.start, t, r.end))
    }
}
