//! Atom selection expressions.
//!
//! The selection parser operates on the small [`AtomLike`] trait instead of a
//! concrete topology type, so it can be used with any atom representation.

use std::fmt;

/// The atom attributes needed by [`Selection`].
pub trait AtomLike {
    /// Atom name (for example, `CA`).
    fn name(&self) -> &str;
    /// Residue name (for example, `ALA`).
    fn resname(&self) -> &str;
    /// Residue identifier. Negative residue identifiers are valid.
    fn resid(&self) -> i32;
    /// Zero-based atom index.
    fn index(&self) -> usize;
    /// Chemical element (for example, `C`), when one is available.
    fn element(&self) -> Option<&str>;
    /// Chain identifier.
    fn chain_id(&self) -> &str;
    /// Segment identifier.
    fn segid(&self) -> &str;
}

/// Errors produced while lexing or parsing a selection expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// The input contained no expression.
    EmptyExpression,
    /// An unexpected character was found at a byte offset.
    UnexpectedCharacter { position: usize, character: char },
    /// The parser reached the end of input while expecting another token.
    UnexpectedEnd { context: &'static str },
    /// A token was not valid in the current position.
    UnexpectedToken { position: usize, token: String },
    /// A predicate name was not supported.
    UnknownPredicate(String),
    /// A predicate value had the wrong form.
    InvalidValue { predicate: String, value: String },
    /// A range was written backwards.
    InvalidRange { start: i64, end: i64 },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExpression => f.write_str("selection expression is empty"),
            Self::UnexpectedCharacter {
                position,
                character,
            } => write!(f, "unexpected character {character:?} at byte {position}"),
            Self::UnexpectedEnd { context } => {
                write!(f, "unexpected end of selection while parsing {context}")
            }
            Self::UnexpectedToken { position, token } => {
                write!(f, "unexpected token {token:?} at byte {position}")
            }
            Self::UnknownPredicate(predicate) => {
                write!(f, "unknown selection predicate {predicate:?}")
            }
            Self::InvalidValue { predicate, value } => {
                write!(f, "invalid value {value:?} for predicate {predicate:?}")
            }
            Self::InvalidRange { start, end } => {
                write!(f, "selection range cannot descend from {start} to {end}")
            }
        }
    }
}

impl std::error::Error for SelectionError {}

/// A parsed selection expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    expression: Expr,
}

impl Selection {
    /// Parse a selection expression.
    pub fn parse(input: &str) -> Result<Self, SelectionError> {
        let tokens = Lexer::new(input).lex()?;
        if tokens.is_empty() {
            return Err(SelectionError::EmptyExpression);
        }

        let mut parser = Parser::new(tokens);
        let expression = parser.parse_expression()?;
        if let Some(token) = parser.peek() {
            return Err(SelectionError::UnexpectedToken {
                position: token.position(),
                token: token.display(),
            });
        }
        Ok(Self { expression })
    }

    /// Test one atom against this selection.
    pub fn matches<A: AtomLike>(&self, atom: &A) -> bool {
        self.expression.matches(atom)
    }

    /// Apply this selection to a slice, preserving the input order.
    pub fn apply<'a, A: AtomLike>(&self, atoms: &'a [A]) -> Vec<&'a A> {
        atoms.iter().filter(|atom| self.matches(*atom)).collect()
    }

    /// Alias for [`Selection::apply`].
    pub fn select<'a, A: AtomLike>(&self, atoms: &'a [A]) -> Vec<&'a A> {
        self.apply(atoms)
    }
}

/// Parse `expression` and return matching atoms in input order.
pub fn select<'a, A: AtomLike>(
    atoms: &'a [A],
    expression: &str,
) -> Result<Vec<&'a A>, SelectionError> {
    Selection::parse(expression).map(|selection| selection.apply(atoms))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Expr {
    All,
    None,
    Predicate(Predicate),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
}

impl Expr {
    fn matches<A: AtomLike>(&self, atom: &A) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Predicate(predicate) => predicate.matches(atom),
            Self::And(left, right) => left.matches(atom) && right.matches(atom),
            Self::Or(left, right) => left.matches(atom) || right.matches(atom),
            Self::Not(expression) => !expression.matches(atom),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Predicate {
    Name(String),
    Resname(String),
    Resid(IntRange),
    Index(IntRange),
    Element(String),
    ChainId(String),
    Segid(String),
}

impl Predicate {
    fn matches<A: AtomLike>(&self, atom: &A) -> bool {
        match self {
            Self::Name(value) => atom.name() == value,
            Self::Resname(value) => atom.resname() == value,
            Self::Resid(range) => range.contains(i64::from(atom.resid())),
            Self::Index(range) => i64::try_from(atom.index())
                .map(|index| range.contains(index))
                .unwrap_or(false),
            Self::Element(value) => atom.element().is_some_and(|element| element == value),
            Self::ChainId(value) => atom.chain_id() == value,
            Self::Segid(value) => atom.segid() == value,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntRange {
    start: i64,
    end: i64,
}

impl IntRange {
    fn contains(self, value: i64) -> bool {
        self.start <= value && value <= self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String, usize),
    Number(i64, usize),
    String(String, usize),
    LParen(usize),
    RParen(usize),
    Colon(usize),
    Dash(usize),
}

impl Token {
    fn position(&self) -> usize {
        match self {
            Self::Ident(_, position)
            | Self::Number(_, position)
            | Self::String(_, position)
            | Self::LParen(position)
            | Self::RParen(position)
            | Self::Colon(position)
            | Self::Dash(position) => *position,
        }
    }

    fn display(&self) -> String {
        match self {
            Self::Ident(value, _) => value.clone(),
            Self::Number(value, _) => value.to_string(),
            Self::String(value, _) => format!("\"{value}\""),
            Self::LParen(_) => "(".to_owned(),
            Self::RParen(_) => ")".to_owned(),
            Self::Colon(_) => ":".to_owned(),
            Self::Dash(_) => "-".to_owned(),
        }
    }
}

struct Lexer<'a> {
    chars: std::str::CharIndices<'a>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.char_indices(),
        }
    }

    fn lex(mut self) -> Result<Vec<Token>, SelectionError> {
        let mut tokens = Vec::new();
        while let Some((position, character)) = self.chars.next() {
            if character.is_whitespace() {
                continue;
            }
            let token = match character {
                '(' => Token::LParen(position),
                ')' => Token::RParen(position),
                ':' => Token::Colon(position),
                '-' => Token::Dash(position),
                '\'' | '"' => self.lex_string(position, character)?,
                character if character.is_ascii_digit() => self.lex_number(position, character)?,
                character if is_identifier_start(character) => {
                    self.lex_identifier(position, character)
                }
                character => {
                    return Err(SelectionError::UnexpectedCharacter {
                        position,
                        character,
                    });
                }
            };
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn lex_number(&mut self, position: usize, first: char) -> Result<Token, SelectionError> {
        let mut value = String::from(first);
        while let Some((_, character)) = self.chars.clone().next() {
            if !character.is_ascii_digit() {
                break;
            }
            value.push(character);
            self.chars.next();
        }
        let number = value
            .parse::<i64>()
            .map_err(|_| SelectionError::InvalidValue {
                predicate: "number".to_owned(),
                value,
            })?;
        Ok(Token::Number(number, position))
    }

    fn lex_identifier(&mut self, position: usize, first: char) -> Token {
        let mut value = String::from(first);
        while let Some((_, character)) = self.chars.clone().next() {
            if !is_identifier_continue(character) {
                break;
            }
            value.push(character);
            self.chars.next();
        }
        Token::Ident(value, position)
    }

    fn lex_string(&mut self, position: usize, quote: char) -> Result<Token, SelectionError> {
        let mut value = String::new();
        while let Some((_, character)) = self.chars.next() {
            if character == quote {
                return Ok(Token::String(value, position));
            }
            if character == '\\' {
                let Some((_, escaped)) = self.chars.next() else {
                    return Err(SelectionError::UnexpectedEnd {
                        context: "quoted value",
                    });
                };
                value.push(escaped);
            } else {
                value.push(character);
            }
        }
        Err(SelectionError::UnexpectedEnd {
            context: "quoted value",
        })
    }
}

fn is_identifier_start(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '*' | '.' | '+' | '/')
}

fn is_identifier_continue(character: char) -> bool {
    is_identifier_start(character) || matches!(character, '#' | '@' | '%')
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, cursor: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.cursor).cloned();
        self.cursor += usize::from(token.is_some());
        token
    }

    fn parse_expression(&mut self) -> Result<Expr, SelectionError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, SelectionError> {
        let mut expression = self.parse_and()?;
        while self.consume_keyword("or") {
            expression = Expr::Or(Box::new(expression), Box::new(self.parse_and()?));
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> Result<Expr, SelectionError> {
        let mut expression = self.parse_unary()?;
        while self.consume_keyword("and") {
            expression = Expr::And(Box::new(expression), Box::new(self.parse_unary()?));
        }
        Ok(expression)
    }

    fn parse_unary(&mut self) -> Result<Expr, SelectionError> {
        if self.consume_keyword("not") {
            return Ok(Expr::Not(Box::new(self.parse_unary()?)));
        }
        if matches!(self.peek(), Some(Token::LParen(_))) {
            let _ = self.next();
            let expression = self.parse_expression()?;
            match self.next() {
                Some(Token::RParen(_)) => return Ok(expression),
                Some(token) => {
                    return Err(SelectionError::UnexpectedToken {
                        position: token.position(),
                        token: token.display(),
                    });
                }
                None => {
                    return Err(SelectionError::UnexpectedEnd {
                        context: "closing parenthesis",
                    });
                }
            }
        }
        self.parse_predicate()
    }

    fn parse_predicate(&mut self) -> Result<Expr, SelectionError> {
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "predicate",
            });
        };
        let predicate = match token {
            Token::Ident(value, _) => value,
            token => {
                return Err(SelectionError::UnexpectedToken {
                    position: token.position(),
                    token: token.display(),
                });
            }
        };
        match predicate.to_ascii_lowercase().as_str() {
            "all" => Ok(Expr::All),
            "none" => Ok(Expr::None),
            "name" => Ok(Expr::Predicate(Predicate::Name(
                self.parse_string_value("name")?,
            ))),
            "resname" => Ok(Expr::Predicate(Predicate::Resname(
                self.parse_string_value("resname")?,
            ))),
            "element" => Ok(Expr::Predicate(Predicate::Element(
                self.parse_string_value("element")?,
            ))),
            "chainid" => Ok(Expr::Predicate(Predicate::ChainId(
                self.parse_string_value("chainID")?,
            ))),
            "segid" => Ok(Expr::Predicate(Predicate::Segid(
                self.parse_string_value("segid")?,
            ))),
            "resid" => Ok(Expr::Predicate(Predicate::Resid(
                self.parse_range("resid")?,
            ))),
            "index" => Ok(Expr::Predicate(Predicate::Index(
                self.parse_range("index")?,
            ))),
            _ => Err(SelectionError::UnknownPredicate(predicate)),
        }
    }

    fn parse_string_value(&mut self, predicate: &str) -> Result<String, SelectionError> {
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "predicate value",
            });
        };
        match token {
            Token::Ident(value, _) | Token::String(value, _) => Ok(value),
            Token::Number(number, _) => Ok(number.to_string()),
            token => Err(SelectionError::InvalidValue {
                predicate: predicate.to_owned(),
                value: token.display(),
            }),
        }
    }

    fn parse_range(&mut self, predicate: &str) -> Result<IntRange, SelectionError> {
        let start = self.parse_signed_integer(predicate)?;
        let end = if self.consume_range_separator() {
            self.parse_signed_integer(predicate)?
        } else {
            start
        };
        if start > end {
            return Err(SelectionError::InvalidRange { start, end });
        }
        if predicate == "index" && start < 0 {
            return Err(SelectionError::InvalidValue {
                predicate: predicate.to_owned(),
                value: start.to_string(),
            });
        }
        Ok(IntRange { start, end })
    }

    fn parse_signed_integer(&mut self, predicate: &str) -> Result<i64, SelectionError> {
        let negative = matches!(self.peek(), Some(Token::Dash(_)));
        if negative {
            let _ = self.next();
        }
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "numeric value",
            });
        };
        let Token::Number(number, _) = token else {
            return Err(SelectionError::InvalidValue {
                predicate: predicate.to_owned(),
                value: token.display(),
            });
        };
        if negative {
            number
                .checked_neg()
                .ok_or_else(|| SelectionError::InvalidValue {
                    predicate: predicate.to_owned(),
                    value: format!("-{number}"),
                })
        } else {
            Ok(number)
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        let Some(Token::Ident(value, _)) = self.peek() else {
            return false;
        };
        if value.eq_ignore_ascii_case(keyword) {
            let _ = self.next();
            true
        } else {
            false
        }
    }

    fn consume_range_separator(&mut self) -> bool {
        if matches!(self.peek(), Some(Token::Colon(_) | Token::Dash(_))) {
            let _ = self.next();
            return true;
        }
        self.consume_keyword("to")
    }
}

#[cfg(test)]
mod tests {
    use super::{AtomLike, Selection, SelectionError, select};

    #[derive(Debug)]
    struct TestAtom {
        name: &'static str,
        resname: &'static str,
        resid: i32,
        index: usize,
        element: Option<&'static str>,
        chain_id: &'static str,
        segid: &'static str,
    }

    impl AtomLike for TestAtom {
        fn name(&self) -> &str {
            self.name
        }
        fn resname(&self) -> &str {
            self.resname
        }
        fn resid(&self) -> i32 {
            self.resid
        }
        fn index(&self) -> usize {
            self.index
        }
        fn element(&self) -> Option<&str> {
            self.element
        }
        fn chain_id(&self) -> &str {
            self.chain_id
        }
        fn segid(&self) -> &str {
            self.segid
        }
    }

    fn atoms() -> Vec<TestAtom> {
        vec![
            TestAtom {
                name: "CA",
                resname: "ALA",
                resid: 1,
                index: 0,
                element: Some("C"),
                chain_id: "A",
                segid: "PROT",
            },
            TestAtom {
                name: "N",
                resname: "ALA",
                resid: 1,
                index: 1,
                element: Some("N"),
                chain_id: "A",
                segid: "PROT",
            },
            TestAtom {
                name: "OW",
                resname: "HOH",
                resid: 8,
                index: 2,
                element: Some("O"),
                chain_id: "B",
                segid: "WAT",
            },
        ]
    }

    #[test]
    fn predicates_and_ranges_match_atoms() {
        let atoms = atoms();
        for (expression, expected) in [
            ("all", 3),
            ("none", 0),
            ("name CA", 1),
            ("resname ALA", 2),
            ("resid 1-1", 2),
            ("index 1:2", 2),
            ("element O", 1),
            ("chainID B", 1),
            ("segid PROT", 2),
        ] {
            let selection = Selection::parse(expression).unwrap();
            assert_eq!(selection.apply(&atoms).len(), expected, "{expression}");
        }
        assert_eq!(select(&atoms, "name CA").unwrap().len(), 1);
    }

    #[test]
    fn boolean_precedence_and_parentheses_work() {
        let atoms = atoms();
        let selection = Selection::parse("name CA or name N and resname HOH").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);

        let selection = Selection::parse("(name CA or name N) and not chainID B").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 2);
    }

    #[test]
    fn quoted_values_and_negative_resids_work() {
        let mut atoms = atoms();
        atoms[0].resid = -2;
        let selection = Selection::parse("resid -2--1").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);

        let selection = Selection::parse("name 'CA'").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);
    }

    #[test]
    fn invalid_expressions_are_reported() {
        assert_eq!(Selection::parse(""), Err(SelectionError::EmptyExpression));
        assert!(matches!(
            Selection::parse("wat CA"),
            Err(SelectionError::UnknownPredicate(_))
        ));
        assert!(matches!(
            Selection::parse("index -1"),
            Err(SelectionError::InvalidValue { .. })
        ));
        assert!(matches!(
            Selection::parse("resid 5-1"),
            Err(SelectionError::InvalidRange { .. })
        ));
        assert!(Selection::parse("(name CA").is_err());
    }
}
