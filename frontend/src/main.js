// Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
// This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
// See LICENSE for details.

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import en from 'element-plus/dist/locale/en.mjs'

import App from './App.vue'
import router from './router'
import i18n, { getLocale } from './i18n'

const elementLocales = {
  'zh-CN': zhCn,
  en,
}

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(i18n)
app.use(ElementPlus, { locale: elementLocales[getLocale()] })

app.mount('#app')
