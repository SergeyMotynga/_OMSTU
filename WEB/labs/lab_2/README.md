# Лабораторная работа №2 - Vue.js SPA

SPA приложение на Vue.js с переиспользуемым компонентом таблицы и роутингом.

## Структура проекта

```
lab_2/
├── src/
│   ├── components/
│   │   └── TableComponent.vue    # Переиспользуемый компонент таблицы
│   ├── views/
│   │   ├── LaureatsView.vue        # Страница лауреатов
│   │   └── PrizesView.vue          # Страница премий
│   ├── router/
│   │   └── index.js                # Конфигурация Vue Router
│   ├── assets/
│   │   └── styles.css              # Стили приложения
│   ├── App.vue                     # Главный компонент
│   └── main.js                     # Точка входа приложения
├── index.html                      # HTML шаблон
├── package.json                    # Зависимости проекта
├── vite.config.js                  # Конфигурация Vite
└── README.md                       # Документация
```

## Установка и запуск

1. Установите зависимости:
```bash
npm install
```

2. Запустите dev-сервер:
```bash
npm run dev
```

3. Соберите проект для продакшена:
```bash
npm run build
```

## Особенности реализации

- **Vue 3** с Composition API
- **Vue Router** для навигации между страницами
- **Переиспользуемый компонент таблицы** (`TableComponent.vue`)
- **Два маршрута**: `/laureats` и `/prizes`
- **SPA** - без перезагрузки страницы при навигации




